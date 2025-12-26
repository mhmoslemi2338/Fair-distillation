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


from utils import orthogonality_loss_from_group_grads, _sanitize_grads

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,

    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1, help='epochs to train a model with synthetic data')

    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--FairDD', action='store_true', help='Enable FairDD')


    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False
    args.FairDD = True
    # args.FairDD = False

    # if not os.path.exists(args.data_path):
    #     os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    # eval_it_pool = [0, 1000]
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
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        color_all = [dst_train[i][2] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        color_all = torch.tensor(color_all, dtype=torch.long, device=args.device)

        args.num_classes = len(torch.unique(labels_all))
        args.num_groups = len(torch.unique(color_all))

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle], labels_all[idx_shuffle], color_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        color_syn = torch.zeros_like(label_syn)
        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_data, _, color_data = get_images(c, args.ipc)
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = image_data.detach().data
                color_syn.data[c * args.ipc:(c + 1) * args.ipc] = color_data.detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    max_Equalized_Odds_list = []
                    mean_Equalized_Odds_list = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test, max_Equalized_Odds, mean_Equalized_Odds, max_Sufficiency, mean_Sufficiency = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                        max_Equalized_Odds_list.append(max_Equalized_Odds)
                        mean_Equalized_Odds_list.append(mean_Equalized_Odds)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    print('accs, max_Equalized_Odds, mean_Equalized_Odds', np.mean(accs),np.mean(max_Equalized_Odds_list), np.mean(mean_Equalized_Odds_list))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''

                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d_Fair.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.


            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.


            # for ol in range(args.outer_loop):
            for ol in range(args.outer_loop):
                

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer


                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                L= 0 
                L2 = 0
                for c in range(num_classes):
                    img_real, label, color = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    if args.FairDD==True:
                    # if False: #exp1
                        # 1. Compute Synthetic Gradients
                        output_syn = net(img_syn)
                        loss_syn = criterion(output_syn, lab_syn)
                        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                        # 2. Compute Real Output (THIS WAS MISSING)
                        output_real = net(img_real)

                        # --- PREPARE REAL GRADIENTS (Handle Imbalance Here) ---
                        unique_groups = torch.unique(color)
                        group_grads = {}
                        
                        # Iterate over each group present in the current batch
                        for grp_idx in unique_groups:
                            mask = (color == grp_idx)
                            if mask.sum() == 0: continue
                            
                            # Calculate average gradient for THIS group only
                            # We use output_real[mask] which is now defined
                            loss_grp = criterion(output_real[mask], lab_real[mask])
                            g_grp = torch.autograd.grad(loss_grp, net_parameters, retain_graph=True)
                            group_grads[grp_idx.item()] = list((_.detach().clone() for _ in g_grp))

                        # 3. Construct BALANCED Real Gradient Target
                        gw_real_balanced = []
                        if len(group_grads) > 0:
                            for i in range(len(gw_syn)): # Iterate over layers
                                # Stack gradients from all groups for this layer
                                layer_grads = [group_grads[k][i] for k in group_grads]
                                # Take the mean across groups (giving them equal weight)
                                gw_real_balanced.append(torch.stack(layer_grads).mean(dim=0))
                        else:
                            # Fallback: if no groups found (rare), use standard full-batch gradient
                            loss_real = criterion(output_real, lab_real)
                            gw_real_fallback = torch.autograd.grad(loss_real, net_parameters, retain_graph=True)
                            gw_real_balanced = list((_.detach().clone() for _ in gw_real_fallback))

                        # 4. Calculate Task Fidelity Loss (Using Balanced Target)
                        loss += match_loss(gw_syn, gw_real_balanced, args)

                        # 5. Calculate Orthogonality/Fairness Constraint
                        if len(unique_groups) > 1:
                            ortho_loss = torch.tensor(0.0).to(args.device)
                            groups_list = list(group_grads.keys())
                            
                            for i in range(len(groups_list)):
                                for j in range(i + 1, len(groups_list)):
                                    g_a = group_grads[groups_list[i]]
                                    g_b = group_grads[groups_list[j]]
                                    
                                    dot_prod = torch.tensor(0.0).to(args.device)
                                    for k in range(len(gw_syn)):
                                        # Difference vector (direction of bias)
                                        diff = g_a[k] - g_b[k]
                                        # Accumulate dot product
                                        dot_prod += torch.sum(gw_syn[k] * diff)
                                    
                                    ortho_loss += dot_prod ** 2
                            
                            fair_lambda = 0.0005
                            loss += fair_lambda * ortho_loss


                    # if args.FairDD==True:
                    #     # 1. Compute Synthetic Gradients
                    #     output_syn = net(img_syn)
                    #     loss_syn = criterion(output_syn, lab_syn)
                    #     gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    #     # 2. Compute Real Output (THIS WAS MISSING)
                    #     output_real = net(img_real)

                    #     # --- PREPARE REAL GRADIENTS (Handle Imbalance Here) ---
                    #     unique_groups = torch.unique(color)
                    #     group_grads = {}
                        
                    #     # Iterate over each group present in the current batch
                    #     for grp_idx in unique_groups:
                    #         mask = (color == grp_idx)
                    #         if mask.sum() == 0: continue
                            
                    #         # Calculate average gradient for THIS group only
                    #         # We use output_real[mask] which is now defined
                    #         loss_grp = criterion(output_real[mask], lab_real[mask])
                    #         g_grp = torch.autograd.grad(loss_grp, net_parameters, retain_graph=True)
                    #         group_grads[grp_idx.item()] = list((_.detach().clone() for _ in g_grp))


                    #     # 5. Calculate Orthogonality/Fairness Constraint
                    #     if len(unique_groups) > 1:
                    #         ortho_loss = torch.tensor(0.0).to(args.device)
                    #         groups_list = list(group_grads.keys())
                            
                    #         for i in range(len(groups_list)):
                    #             for j in range(i + 1, len(groups_list)):
                    #                 g_a = group_grads[groups_list[i]]
                    #                 g_b = group_grads[groups_list[j]]
                                    
                    #                 dot_prod = torch.tensor(0.0).to(args.device)
                    #                 for k in range(len(gw_syn)):
                    #                     # Difference vector (direction of bias)
                    #                     diff = g_a[k] - g_b[k]
                    #                     # Accumulate dot product
                    #                     dot_prod += torch.sum(gw_syn[k] * diff)
                                    
                    #                 ortho_loss += dot_prod ** 2



                            
                    #         loss_real = criterion(output_real, lab_real)
                    #         gw_real = torch.autograd.grad(loss_real, net_parameters)
                    #         gw_real = list((_.detach().clone() for _ in gw_real))

                    #         output_syn = net(img_syn)
                    #         loss_syn = criterion(output_syn, lab_syn)
                    #         gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                            


                    #         fair_lambda = 0.0005
                    #         loss += match_loss(gw_syn, gw_real, args)
                    #         loss += fair_lambda * ortho_loss



                    else:
                        output_real = net(img_real)
                        loss_real = criterion(output_real, lab_real)
                        gw_real = torch.autograd.grad(loss_real, net_parameters)
                        gw_real = list((_.detach().clone() for _ in gw_real))

                        output_syn = net(img_syn)
                        loss_syn = criterion(output_syn, lab_syn)
                        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                        loss += match_loss(gw_syn, gw_real, args)

                optimizer_img.zero_grad()
                loss.backward()
                
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == args.outer_loop - 1:
                    break


                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)





            loss_avg /= (num_classes*args.outer_loop)

            # L /= (num_classes*args.outer_loop)
            # L2 /= (num_classes*args.outer_loop)
            # print(L, L2)




            # if it%10 == 0:


            print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))


            # save_every = max(1, args.Iteration // 10)
            # if it % save_every == 0 or it == args.Iteration:
            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                # data_save = ([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dip%d.pt'%(args.method, args.dataset, args.model, args.ipc,args.Iteration)))
                print('save synthetic data to %s'%(os.path.join(args.save_path, 'res_%s_%s_%s_%dip.pt'%(args.method, args.dataset, args.model, args.ipc))))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



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

