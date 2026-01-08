import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from math import ceil

# Imports from the local DC repository utils
from utils import get_dataset, get_network, evaluate_synset, get_time, DiffAugment, ParamDiffAug, TensorDataset, match_loss
from new_strategy import NEW_Strategy

def get_images(images_all, indices_class, c, n):
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle]

class Synthesizer():
    """Condensed data class
    """
    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device

        # Initialize data in [0, 1] range (Raw image space)
        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)
        
        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, images_all, indices_class, model, mean, std, init_type='noise'):
        """Condensed data initialization
        """
        # Helper to denormalize real images before storing them in Synthesizer (which expects [0,1])
        def denormalize(img):
            mean_t = torch.tensor(mean).to(self.device).view(1, -1, 1, 1)
            std_t = torch.tensor(std).to(self.device).view(1, -1, 1, 1)
            return img * std_t + mean_t

        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass):
                img = get_images(images_all, indices_class, c, self.ipc)
                img = denormalize(img)
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            print("Mixed initialize synset")
            for c in range(self.nclass):
                img = get_images(images_all, indices_class, c, self.ipc * self.factor**2)
                img = denormalize(img)
                img = img.data.to(self.device)

                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc

                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(img[k * n:(k + 1) * n], size=(h_r, w_r))
                        self.data.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                                       w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == 'noise':
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def decode_zoom(self, img, target, factor):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor**2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode_zoom_multi(self, img, target, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, img, target, factor_max, bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor**2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = self.decode_zoom(img[idx:idx + n], target[idx:idx + n], factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(self, data, target, bound=128):
        """Multi-formation
        """
        if self.factor > 1:
            if self.decode_type == 'multi':
                data, target = self.decode_zoom_multi(data, target, self.factor)
            elif self.decode_type == 'bound':
                data, target = self.decode_zoom_bound(data, target, self.factor, bound=bound)
            else:
                data, target = self.decode_zoom(data, target, self.factor)

        return data, target

    def sample(self, c, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target = self.subsample(data, target, max_size=max_size)
        return data, target


def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

    return dist_

def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss

def condense(args):
    # Setup Device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Dataset (Using DC utils)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    
    # Organize real dataset
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    print("Building dataset...")
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]

    color_all = [dst_train[i][2] for i in range(len(dst_train))]
    color_all = torch.tensor(color_all, dtype=torch.long, device=args.device)




    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
    

    args.num_classes = len(torch.unique(labels_all))
    args.num_groups = len(torch.unique(color_all))

    # Define Normalizer for Syn Data (Syn data is [0,1], Model expects Normalized)
    def normalize_t(img):
        mean_t = torch.tensor(mean).to(args.device).view(1, -1, 1, 1)
        std_t = torch.tensor(std).to(args.device).view(1, -1, 1, 1)
        return (img - mean_t) / std_t
    
    # Define Denormalizer (for initialization from real data)
    def denormalize_t(img):
        mean_t = torch.tensor(mean).to(args.device).view(1, -1, 1, 1)
        std_t = torch.tensor(std).to(args.device).view(1, -1, 1, 1)
        return img * std_t + mean_t

    # Initialize Synthesizer
    synset = Synthesizer(args, num_classes, channel, im_size[0], im_size[1], device=args.device)
    
    # Initialize Model
    model = get_network(args.model, channel, num_classes, im_size).to(args.device)
    model.eval()

    # Initialization Strategy
    if args.init == 'kmean':
        print("Kmean initialize synset")
        for c in range(synset.nclass):
            img = get_images(images_all, indices_class, c, len(indices_class[c])) # Get all images
            strategy = NEW_Strategy(img, model)
            query_idxs = strategy.query(args.ipc)
            img_init = img[query_idxs].detach()
            # De-normalize before storing
            synset.data.data[c*synset.ipc:(c+1)*synset.ipc] = denormalize_t(img_init).data
            
    elif args.init == 'random':
        synset.init(images_all, indices_class, model, mean, std, init_type='random')
        
    elif args.init == 'mix':
        # Custom mix init using NEW_Strategy if f2_init is not random
        if getattr(args, 'f2_init', 'random') == 'random':
            synset.init(images_all, indices_class, model, mean, std, init_type='mix')
        else:
             print("Mixed initialize synset (Smart)")
             for c in range(synset.nclass):
                img = get_images(images_all, indices_class, c, len(indices_class[c]))
                strategy = NEW_Strategy(img, model)
                query_idxs = strategy.query(synset.ipc * synset.factor**2)
                img = img[query_idxs].detach()
                img = denormalize_t(img) # Denormalize
                
                # Copy-paste Mix Logic
                s = synset.size[0] // synset.factor
                remained = synset.size[0] % synset.factor
                k = 0
                n = synset.ipc
                h_loc = 0
                for i in range(synset.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(synset.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(img[k * n:(k + 1) * n], size=(h_r, w_r))
                        synset.data.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r, w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

    # Query list for inner loop selection
    query_list = torch.tensor(np.ones(shape=(num_classes, args.batch_real)), dtype=torch.long, requires_grad=False, device=args.device)

    # Optimization
    optim_img = torch.optim.SGD(synset.parameters(), lr=args.lr_img, momentum=args.mom_img)
    
    print(f"Start condensing with {args.match} matching...")
    
    # Outer Loop
    for it in range(args.Iteration + 1):
        
        # Periodic Model Reset
        if it % args.fix_iter == 0 and it != 0:
            model = get_network(args.model, channel, num_classes, im_size).to(args.device)
            model.train()
            # In DREAM, they might pretrain. Here we just reset.
            # If pretrain is needed, load_ckpt logic goes here.

        loss_total = 0
        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
        
        for ot in range(args.inner_loop):
            for c in range(num_classes):
                # Update Query List (Smart Sampling)
                if ot % args.interval == 0:
                    img = get_images(images_all, indices_class, c, len(indices_class[c]))
                    strategy = NEW_Strategy(img, model)
                    query_idxs = strategy.query(args.batch_real)
                    query_list[c] = query_idxs

                # Get Real Data
                img_real_all = get_images(images_all, indices_class, c, len(indices_class[c]))
                img_real = img_real_all[query_list[c]] # Normalized
                lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                
                # Get Syn Data
                img_syn, lab_syn = synset.sample(c, max_size=args.batch_syn_max)
                img_syn = normalize_t(img_syn) # Normalize for model

                # Augmentation
                # DiffAugment expects normalized images usually, DC utils handles it.
                if args.dsa_strategy != 'none':
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                # Matching Loss
                loss = None
                
                if args.match == 'feat':
                    # Feature Matching using model.embed (DC standard)
                    # Note: DREAM original uses get_feature with index range. 
                    # Here we use the global embed output.
                    feat_real = model.embed(img_real).detach()
                    feat_syn = model.embed(img_syn)
                    loss = dist(feat_real.mean(0), feat_syn.mean(0), method=args.metric)

                elif args.match == 'grad':
                    # Gradient Matching
                    criterion = nn.CrossEntropyLoss()
                    
                    output_real = model(img_real)
                    loss_real = criterion(output_real, lab_real)
                    g_real = torch.autograd.grad(loss_real, model.parameters())
                    g_real = list((g.detach() for g in g_real))

                    output_syn = model(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

                    # Use DC utils match_loss logic or custom loop
                    for i in range(len(g_real)):
                         if (len(g_real[i].shape) == 1) and not args.bias: continue # bias, norm
                         if (len(g_real[i].shape) == 2) and not args.fc: continue
                         loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))

                loss_total += loss.item()
                optim_img.zero_grad()
                loss.backward()
                optim_img.step()

        # if it % 100 == 0:
        if it % 2 == 0:
            print(f"Iter {it}: Loss {loss_total:.4f}")

        # Evaluation
        if it == args.Iteration-1:
        # if False:
            print("Evaluating...")
            # Create a full normalized synthetic dataset for evaluation
            data_dec = []
            target_dec = []
            with torch.no_grad():
                for c in range(num_classes):
                    idx_from = synset.ipc * c
                    idx_to = synset.ipc * (c + 1)
                    data = synset.data[idx_from:idx_to].detach()
                    target = synset.targets[idx_from:idx_to].detach()
                    data, target = synset.decode(data, target)
                    data_dec.append(data)
                    target_dec.append(target)
            
            data_dec = torch.cat(data_dec)
            data_dec = normalize_t(data_dec) # Normalize!
            target_dec = torch.cat(target_dec)
            
            # Save Image (Unnormalized for visualization)
            save_image(denormalize_t(data_dec[:min(100, len(data_dec))]), os.path.join(args.save_path, f'img{it}.png'))
            
            # Evaluate using DC utils
            # Note: DC evaluate_synset trains a new network from scratch on syn data
            # We need to temporarily mock the standard DC training loop expectation
            # evaluate_synset expects tensors
            net_eval = get_network(args.model, channel, num_classes, im_size).to(args.device)
            _, acc_train, acc_test, EOD_max, EOD_mean,a,b = evaluate_synset(it, net_eval, data_dec, target_dec, testloader, args)
            print(f"Iter {it}: Test Acc {acc_test:.4f}")
            
            torch.save([synset.data.detach().cpu(), synset.targets.cpu()], os.path.join(args.save_path, f'data_{it}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DREAM Condensation')
    # Standard DC Args
    parser.add_argument('--dataset', type=str, default='CIFAR10_S_90', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result_dream', help='path to save results')
    parser.add_argument('--Iteration', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=5e-3, help='learning rate for updating synthetic images')
    parser.add_argument('--mom_img', type=float, default=0.5, help='momentum for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    
    # DREAM Specific Args
    parser.add_argument('--factor', type=int, default=2, help='condensation factor')
    parser.add_argument('--decode_type', type=str, default='multi', choices=['multi', 'bound', 'single'], help='multi-formation type')
    parser.add_argument('--init', type=str, default='kmean', choices=['random', 'noise', 'mix', 'kmean'], help='initialization')
    parser.add_argument('--match', type=str, default='grad', choices=['feat', 'grad'], help='matching metric')
    parser.add_argument('--metric', type=str, default='mse', help='distance metric')
    parser.add_argument('--inner_loop', type=int, default=50, help='inner loop')
    parser.add_argument('--interval', type=int, default=10, help='interval for query')
    parser.add_argument('--fix_iter', type=int, default=100, help='interval for model reset')
    parser.add_argument('--batch_syn_max', type=int, default=128, help='max synthetic batch size')
    parser.add_argument('--bias', action='store_true', help='match bias')
    parser.add_argument('--fc', action='store_true', help='match fc layer')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')

    args = parser.parse_args()
    args.bias = False
    args.fc = False

    #             args.lr_img = 
#             args.n_data = 2000
    
    # Setup DiffAug
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.dsa_strategy != 'none' else False
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    condense(args)