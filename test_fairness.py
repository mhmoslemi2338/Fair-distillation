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

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,

    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=35, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')

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
    parser.add_argument('--testMetric', type=str, default='DC', help='distance metric2')


    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False
    args.FairDD = False




    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    load_random_state(random_state)



    for exp in range(args.num_exp):
        # print('\n================== Exp %d ==================\n '%exp)
        # print('Hyper-parameters: \n', args.__dict__)
        # print('Evaluation model pool: ', model_eval_pool)

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

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle], labels_all[idx_shuffle], color_all[idx_shuffle]





        # for name in ['DC','DM','IDC', 'Random','full']:
        for name in ['DC']:
            args.testMetric = name
            # for ipc in [10, 50,100]:
            for ipc in [10]:
                if name == 'full':
                    if ipc in [50,100]: continue

                args.ipc = ipc

                if name == 'Random':
                    image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
                    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
                    for c in range(num_classes):
                        image_data, _, color_data = get_images(c, args.ipc)
                        image_syn.data[c * args.ipc:(c + 1) * args.ipc] = image_data.detach().data

                elif name == 'full':
                    image_syn = copy.deepcopy(images_all)
                    label_syn = copy.deepcopy(labels_all)
                    args.num_eval = 5

                
                else:
                    save_path = '/home/mmoslem3/scratch/FairDD/results/DC-No-Ortho/Fair_NoOrtho_'+args.testMetric+'_' + args.dataset + '_ipc'  + str(args.ipc) +'/res_'+args.testMetric+'_' + args.dataset + '_ConvNet_'+str(args.ipc)+'ip1000.pt' # Your file path
                    save_path = '/home/mmoslem3/scratch/FairDD/result/res_DC_UTKface_ConvNet_10ip1000.pt' # Your file path
                    # save_path = '/home/mmoslem3/scratch/FairDD/results/res_'+args.testMetric+'_' + args.dataset + '_ConvNet_'+str(args.ipc)+'ip100.pt' # Your file path
                    print(save_path)
                    checkpoint = torch.load(save_path, map_location=args.device, weights_only=False)
                    data_list = checkpoint['data']

                    if name == 'IDC':
                        image_syn, label_syn = data_list
                    else:
                        # image_syn, label_syn = data_list[0]
                        image_syn, label_syn = data_list

                image_syn = image_syn.to(args.device)
                label_syn = label_syn.to(args.device)
                

                ''' Evaluate synthetic data '''
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s'%(args.model, model_eval))

                    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                    # print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    max_Equalized_Odds_list = []
                    mean_Equalized_Odds_list = []

                    max_Sufficiency_list = []
                    mean_Sufficiency_list = []

                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test, max_Equalized_Odds, mean_Equalized_Odds, max_Sufficiency, mean_Sufficiency = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                        
                        max_Equalized_Odds_list.append(max_Equalized_Odds)
                        mean_Equalized_Odds_list.append(mean_Equalized_Odds)
                        
                        max_Sufficiency_list.append(max_Sufficiency)
                        mean_Sufficiency_list.append(mean_Sufficiency)
                        print('Evaluation %d '%(it_eval))




                    print('\n\n -------Model = %s, dataset = %s, ipc = %d --------- '%(args.testMetric, args.dataset, args.ipc))
                    # print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    print('Accuracy: %0.6f ± %0.6f \nmax_Equalized_Odds: %0.6f ± %0.6f \nmean_Equalized_Odds: %0.6f ± %0.6f\nmax_Sufficiency: %0.6f ± %0.6f\nmean_Sufficiency: %0.6f ± %0.6f'%(np.mean(accs),np.std(accs),np.mean(max_Equalized_Odds_list),np.std(max_Equalized_Odds_list), np.mean(mean_Equalized_Odds_list),np.std(mean_Equalized_Odds_list), np.mean(max_Sufficiency_list),np.std(max_Sufficiency_list), np.mean(mean_Sufficiency_list),np.std(mean_Sufficiency_list)))
                    print('--------------------------------\n\n')





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

