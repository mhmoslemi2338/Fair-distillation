import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
import random


def mean_moments_diag(F, eps=1e-6):
    # F: [n, d]
    mu = F.mean(dim=0)
    X  = F - mu
    m2 = (X**2).mean(dim=0) + eps
    m3 = (X**3).mean(dim=0)
    m4 = (X**4).mean(dim=0)

    # standardized (more stable across scale)
    skew = m3 / (m2.sqrt()**3 + eps)
    kurt = m4 / (m2**2 + eps)         # use "excess kurtosis" = kurt - 3 if you want
    return mu, m2, skew, kurt


def mmd_rbf(x, y, sigma=1.0):
    # x: [n,d], y: [m,d]
    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    kxx = torch.exp(-xx / (2 * sigma**2))
    kyy = torch.exp(-yy / (2 * sigma**2))
    kxy = torch.exp(-xy / (2 * sigma**2))
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()

def mean_var(F, eps=1e-6):
    # F: [n, d]
    mu = F.mean(dim=0)
    var = F.var(dim=0, unbiased=False) + eps
    return mu, var


def mean_cov(F, eps=1e-4, shrink=0.05):
    """
    F: [n, k] features in projected space
    returns mu: [k], C: [k,k] SPD covariance (shrinkage + epsI)
    """
    n, k = F.shape
    mu = F.mean(dim=0)
    X = F - mu
    denom = max(n - 1, 1)
    C = (X.T @ X) / denom
    I = torch.eye(k, device=F.device, dtype=F.dtype)
    C = (1 - shrink) * C + shrink * I
    C = C + eps * I
    return mu, C

def spd_logm(A, eps=1e-6):
    A = 0.5 * (A + A.T)
    w, V = torch.linalg.eigh(A)
    w = w.clamp_min(eps)
    return (V * w.log()) @ V.T

def spd_expm(A):
    A = 0.5 * (A + A.T)
    w, V = torch.linalg.eigh(A)
    return (V * w.exp()) @ V.T



def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10_S_90', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    # parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    # parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    # parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=1000, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--shuffle', type=bool, default=False, help='distance metric')
    parser.add_argument('--FairDD', action='store_true', help='Enable FairDD')
    parser.add_argument('--group_balance', type=bool, default=False, help='distance metric')



    args = parser.parse_args()
    args.method = 'DM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.FairDD = True


    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = [args.Iteration]
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    # 恢复之前的随机状态
    load_random_state(random_state)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []


    for exp in range(args.num_exp):
        # print('\n================== Exp %d ==================\n '%exp)
        # print('Hyper-parameters: \n', args.__dict__)
        # print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        color_all = []

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [int(dst_train[i][1]) for i in range(len(dst_train))]
        color_all = [int(dst_train[i][2]) for i in range(len(dst_train))]
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        color_all = torch.tensor(color_all, dtype=torch.long, device=args.device)

        args.num_classes = len(torch.unique(labels_all))
        args.num_groups = len(torch.unique(color_all))

        indices_class = [[] for c in range(args.num_classes)]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        # for c in range(num_classes):
        #     print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle], labels_all[idx_shuffle], color_all[idx_shuffle]

        # for ch in range(channel):
        #     print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(args.num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        color_syn = torch.zeros_like(label_syn)
        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(args.num_classes):
                image_data, _, color_data = get_images(c, args.ipc)
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = image_data.detach().data
                color_syn.data[c*args.ipc:(c+1)*args.ipc] = color_data.detach().data
        else:
            print('initialize synthetic data from random noise')

        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        print()
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):
            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    # print('DSA augmentation strategy: \n', args.dsa_strategy)
                    # print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                    accs = []
                    max_Equalized_Odds_list = []
                    mean_Equalized_Odds_list = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, args.num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        # _, acc_train, acc_test, max_Equalized_Odds, mean_Equalized_Odds = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        _, acc_train, acc_test, max_Equalized_Odds, mean_Equalized_Odds, max_Sufficiency, mean_Sufficiency = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                        max_Equalized_Odds_list.append(max_Equalized_Odds)
                        mean_Equalized_Odds_list.append(mean_Equalized_Odds)
                        # torch.save({'net': net_eval.state_dict()}, os.path.join(args.save_path,'res_%s_%s_%s_%sori.pt' % (args.method, args.dataset,args.model,it_eval)))
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    print('\n\naccs, max_Equalized_Odds, mean_Equalized_Odds',np.mean(accs), np.round(np.mean(max_Equalized_Odds_list),4), np.round(np.mean(mean_Equalized_Odds_list),4),'\n\n')

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

            ''' Train synthetic data '''
            net = get_network(args.model, channel, args.num_classes, im_size).to(args.device) # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel
            # embed.eval()

            loss_avg = 0


            # # if not hasattr(args, "proj_mat"):
            #     # you need embed output dim; infer once
            # with torch.no_grad():
            #     dummy = torch.zeros(1, channel, im_size[0], im_size[1], device=args.device)
            #     d = embed(dummy).shape[1]
            # k = getattr(args, "cov_k", 64)
            # proj = (torch.randn(d, k, device=args.device) / (k ** 0.5))

            if not hasattr(args, "proj_mat"):
                # you need embed output dim; infer once
                with torch.no_grad():
                    dummy = torch.zeros(1, channel, im_size[0], im_size[1], device=args.device)
                    d = embed(dummy).shape[1]
                k = getattr(args, "cov_k", 128)
                args.proj_mat = (torch.randn(d, k, device=args.device) / (k ** 0.5))

            proj = args.proj_mat


            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(args.device)
            for c in range(args.num_classes):
                img_real, label, color = get_images(c, args.batch_real)
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)


                
                if True:

                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)





                    unique_groups = torch.unique(color_all)
                    group_means = []
                    syn_mean = torch.mean(output_syn, dim=0)

                    
                    for g in unique_groups:
                        mask = (color == g)
                        mu_g = embed(img_real[mask])
                        mu_g = torch.mean(mu_g, dim=0)
                        group_means.append(mu_g)

                    real_barycenter = torch.mean(torch.stack(group_means, dim=0), dim=0)

                    L = real_barycenter - torch.mean(output_syn, dim=0)

                    loss += torch.sum(L.abs())
                    
                    
                
                
                
                if False:
                    # ---- embeddings: forward ONCE ----
                    feat_real = embed(img_real).detach()     # [Br, d]
                    feat_syn = embed(img_syn)                # [Bs, d] keep grad for syn images

                    # ---- balanced barycenter target across groups (single target per class) ----
                    unique_groups = torch.unique(color)
                    mus, vars_ = [], []

                    for g in unique_groups:
                        m = (color == g)
                        mu_g, var_g = mean_var(feat_real[m])
                        mus.append(mu_g)
                        vars_.append(var_g)

                    mu_star  = torch.stack(mus, dim=0).mean(dim=0)     # [d]
                    var_star = torch.stack(vars_, dim=0).mean(dim=0)   # [d]

                    mu_syn, var_syn = mean_var(feat_syn)

                    # lam_var = 0.2
                    # loss += lam_var * torch.sum((var_syn - var_star.detach()).pow(2))
                    # loss += torch.sum((mu_syn - mu_star.detach()).abs())

                    # # iter 3000  44.49 22.025 8.2788 
                    # lam_var = 0.1
                    # loss += lam_var * torch.sum((var_syn - var_star.detach()).abs())
                    # loss += torch.sum((mu_syn - mu_star.detach()).abs())

                    lam_var = 0.05
                     # with 0.06 and 512 and 2000 got 44.78 20.5125 8.1187 
                     # with 0.1 and 512 and 2000 got 44.08875 21.8625 8.5712 
                    loss += lam_var * torch.sum((var_syn - var_star.detach()).abs())
                    loss += torch.sum((mu_syn - mu_star.detach()).abs())


                if False: # iter 1000  43.714999999999996 22.15 8.5125 
                    feat_real = embed(img_real).detach()     # [Br, d]
                    feat_syn = embed(img_syn)         # [Bs,d] keep grad

                    uniq = torch.unique(color)
                    mus, m2s, m3s, m4s = [], [], [], []

                    for g in uniq:
                        m = (color == g)
                        mu_g, m2_g, m3_g, m4_g = mean_moments_diag(feat_real[m])
                        mus.append(mu_g); m2s.append(m2_g); m3s.append(m3_g); m4s.append(m4_g)

                    mu_star = torch.stack(mus).mean(0)
                    m2_star = torch.stack(m2s).mean(0)
                    m3_star = torch.stack(m3s).mean(0)
                    m4_star = torch.stack(m4s).mean(0)

                    mu_syn, m2_syn, m3_syn, m4_syn = mean_moments_diag(feat_syn)


                    lam2  = 0.1
                    lam3  = 0.005
                    lam4  = 0.005

                    loss += torch.sum((mu_syn - mu_star.detach()).abs())
                    loss += lam2 * torch.sum((m2_syn - m2_star.detach()).abs())
                    loss += lam3 * torch.sum((m3_syn - m3_star.detach()).abs())
                    loss += lam4 * torch.sum((m4_syn - m4_star.detach()).abs())

                if False:

                    feat_real = embed(img_real).detach()     # [Br, d]
                    feat_syn = embed(img_syn)          # [Bs, d] keep grad for syn

                    # ---- project to k dims (cheaper full-cov) ----
                    z_real = feat_real @ proj          # [Br, k]
                    z_syn  = feat_syn  @ proj          # [Bs, k]

                    # ---- build ONE balanced target Gaussian across groups ----
                    unique_groups = torch.unique(color)
                    mus, covs = [], []

                    for g in unique_groups:
                        m = (color == g)
                        if m.sum() < 2:
                            continue
                        mu_g, C_g = mean_cov(z_real[m])
                        mus.append(mu_g)
                        covs.append(C_g)

                    if len(covs) == 0:
                        # fallback: class-level stats
                        mu_star, C_star = mean_cov(z_real)
                    else:
                        mu_star = torch.stack(mus, dim=0).mean(dim=0)   # [k]
                        # log-Euclidean SPD mean: exp(mean(log(C_g)))
                        logCs = torch.stack([spd_logm(C) for C in covs], dim=0)  # [G,k,k]
                        C_star = spd_expm(logCs.mean(dim=0))                     # [k,k]

                    # ---- syn stats ----
                    mu_syn, C_syn = mean_cov(z_syn)

                    # ---- single loss for this class (no per-group distances) ----
                    
                    lam_cov = .005
                    loss +=  torch.sum((mu_syn - mu_star.detach()).abs()) # only this, 256, iter 3000, 42.92375 20.2 6.9888 
                    # loss += lam_cov * torch.sum((C_syn  - C_star.detach()).abs())





                # output_real = embed(img_real).detach()
                # output_syn = embed(img_syn)


                # unique_groups = torch.unique(color)
                # group_means = []
                # syn_mean = torch.mean(output_syn, dim=0)

                # for g in unique_groups:
                #     mask = (color == g)
                #     if mask.any():
                #         mu_g = embed(img_real[mask]).mean(dim=0)
                #         group_means.append(mu_g)

                # # if len(group_means) > 0:
                # real_barycenter = torch.stack(group_means, dim=0).mean(dim=0)
                # loss += torch.sum((syn_mean - real_barycenter.detach()).abs())






                # output_real = embed(img_real).detach()
                # output_syn = embed(img_syn)

                # for col in torch.unique(color):
                #     loss += torch.sum((torch.mean(output_real[(color == col)], dim=0) - torch.mean(output_syn, dim=0)) ** 2)


                # output_real = embed(img_real).detach()
                # output_syn = embed(img_syn)
                # loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)




            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()


            loss_avg /= (args.num_classes)

            if it%50 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))





if __name__ == '__main__':
    def save_random_state():
        return {
            'torch': torch.get_rng_state(),
            'np': np.random.get_state(),
            'random': random.getstate(),
            'cuda': torch.cuda.get_rng_state_all()
        }
    def load_random_state(state):
        torch.set_rng_state(state['torch'])
        np.random.set_state(state['np'])
        random.setstate(state['random'])
        torch.cuda.set_rng_state_all(state['cuda'])

    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 保存当前的随机状态
    random_state = save_random_state()

    main()
