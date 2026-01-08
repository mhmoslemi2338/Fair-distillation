# dream.py
# ------------------------------------------------------------
# DREAM runner for the TARAZ codebase
#
# Implements DREAM (ICCV 2023) as a *plug-in strategy* on top of
# TARAZ's gradient-matching condensation loop:
#   1) Clustering-based initialization (per-class sub-clusters)
#   2) Representative Matching (sample real batches evenly from sub-clusters)
#
# It uses YOUR existing dataset handlers in utils.get_dataset(...)
# and YOUR existing networks/match_loss/evaluate_synset utilities.
#
# DATASETS covered (as you requested):
#   "CIFAR10_S_90"
#   "Colored_FashionMNIST_foreground"
#   "Colored_FashionMNIST_background"
#   "Colored_MNIST_foreground"
#   "Colored_MNIST_background"
#   "UTKface"
# ------------------------------------------------------------

import os
import time
import copy
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

# TARAZ utilities (your codebase)
from utils import (
    get_loops,
    get_dataset,
    get_network,
    get_eval_pool,
    evaluate_synset,
    get_daparam,
    match_loss,
    get_time,
    TensorDataset,
    epoch,
    DiffAugment,
    ParamDiffAug,
)

# ---------------------------
# Presets you asked for
# ---------------------------

DATASETS = (
    "CIFAR10_S_90",
    # "Colored_FashionMNIST_foreground",
    # "Colored_FashionMNIST_background",
    # "Colored_MNIST_foreground",
    # "Colored_MNIST_background",
    # "UTKface",
)

# Per-dataset DREAM hyperparams (reasonable defaults; tweak freely)
DATASET_PRESETS: Dict[str, Dict] = {
    "CIFAR10_S_90": dict(ipc=10, iters=300, batch_real=256, batch_train=256, cluster_K=128, rep_per_cluster=2),
    "Colored_FashionMNIST_foreground": dict(ipc=10, iters=200, batch_real=256, batch_train=256, cluster_K=64, rep_per_cluster=2),
    "Colored_FashionMNIST_background": dict(ipc=10, iters=200, batch_real=256, batch_train=256, cluster_K=64, rep_per_cluster=2),
    "Colored_MNIST_foreground": dict(ipc=10, iters=200, batch_real=256, batch_train=256, cluster_K=64, rep_per_cluster=2),
    "Colored_MNIST_background": dict(ipc=10, iters=200, batch_real=256, batch_train=256, cluster_K=64, rep_per_cluster=2),
    "UTKface": dict(ipc=10, iters=400, batch_real=128, batch_train=256, cluster_K=64, rep_per_cluster=2),
}

GLOBAL_PRESET = dict(
    method="DREAM",          # label only
    model="ConvNet",
    eval_mode="SS",          # evaluate on the same arch
    num_exp=1,
    num_eval=1,
    lr_img=0.1,
    lr_net=0.01,
    init="dream",            # dream init = cluster-based; "real"/"noise" are overridden
    dsa=False,               # keep TARAZ DC loop consistent; you can flip to True if desired
    dsa_strategy="None",
    dis_metric="ours",
    FairDD=False,            # your FairDD branch stays available but default off
    data_path="data",
    save_root="result_dream",
    seed=42,
)

# ---------------------------
# Determinism helpers
# ---------------------------

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# DREAM: feature + clustering
# ---------------------------

@torch.no_grad()
def pooled_pixel_features(x: torch.Tensor, out_hw: int = 8) -> torch.Tensor:
    """
    Cheap, dependency-free features for clustering:
      - adaptive avg pool to out_hw x out_hw
      - flatten
    Works for 32x32 and 64x64 and any channel count.
    """
    # x: [N, C, H, W]
    x = F.adaptive_avg_pool2d(x, (out_hw, out_hw))
    return x.flatten(1)  # [N, C*out_hw*out_hw]


@torch.no_grad()
def kmeans_torch(
    X: torch.Tensor,
    K: int,
    iters: int = 20,
    assign_bs: int = 8192,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple k-means on X (NxD) in torch.
    Returns:
      centers: [K, D]
      labels:  [N]
    """
    N, D = X.shape
    device = X.device

    if K <= 1 or N == 0:
        centers = X.mean(dim=0, keepdim=True) if N > 0 else torch.zeros((1, D), device=device, dtype=X.dtype)
        labels = torch.zeros((N,), device=device, dtype=torch.long)
        return centers, labels

    # init centers from random points
    perm = torch.randperm(N, device=device)
    centers = X[perm[:K]].clone()

    for _ in range(iters):
        # assign step (chunked)
        labels = torch.empty((N,), device=device, dtype=torch.long)

        c2 = (centers * centers).sum(dim=1).view(1, K)  # [1,K]
        for s in range(0, N, assign_bs):
            e = min(N, s + assign_bs)
            xb = X[s:e]                                # [B,D]
            x2 = (xb * xb).sum(dim=1).view(-1, 1)       # [B,1]
            # dist^2 = ||x||^2 + ||c||^2 - 2 x·c
            dist2 = x2 + c2 - 2.0 * (xb @ centers.t())  # [B,K]
            labels[s:e] = torch.argmin(dist2, dim=1)

        # update step
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros((K,), device=device, dtype=torch.long)

        new_centers.index_add_(0, labels, X)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.long))

        # handle empties
        empty = counts == 0
        if empty.any():
            # reinit empty centers to random points
            ridx = torch.randperm(N, device=device)[:int(empty.sum().item())]
            new_centers[empty] = X[ridx]
            counts[empty] = 1

        centers = new_centers / (counts.float().unsqueeze(1) + eps)

    return centers, labels


@torch.no_grad()
def build_representatives_per_class(
    images_all: torch.Tensor,
    indices_class: List[List[int]],
    K: int,
    rep_per_cluster: int,
    feat_hw: int = 8,
    kmeans_iters: int = 20,
    device: str = "cuda",
) -> Dict[int, List[List[int]]]:
    """
    For each class c:
      - compute pooled features for all images in class
      - run k-means into K sub-clusters
      - for each cluster pick top rep_per_cluster closest-to-center samples
    Returns:
      reps[c] = list length K, each element is a list[int] of representative indices (global indices into images_all)
    """
    reps: Dict[int, List[List[int]]] = {}

    for c, idxs in enumerate(indices_class):
        idxs = list(map(int, idxs))
        n = len(idxs)
        if n == 0:
            reps[c] = []
            continue

        Kc = min(K, n)
        Xc = images_all[idxs].to(device, non_blocking=True)
        Fc = pooled_pixel_features(Xc, out_hw=feat_hw)  # [n,D]

        centers, labels = kmeans_torch(Fc, K=Kc, iters=kmeans_iters)

        # build per-cluster reps by distance to center
        reps_c: List[List[int]] = [[] for _ in range(Kc)]
        c2 = (centers * centers).sum(dim=1).view(1, Kc)

        # compute distances chunked
        bs = 4096
        all_dist2 = torch.empty((n,), device=device, dtype=Fc.dtype)
        for k in range(Kc):
            # dist to center k only, so we don't store full n x K matrix
            ck = centers[k].view(1, -1)
            ck2 = (ck * ck).sum(dim=1).view(1, 1)
            # chunk
            dists = []
            for s in range(0, n, bs):
                e = min(n, s + bs)
                xb = Fc[s:e]
                x2 = (xb * xb).sum(dim=1, keepdim=True)
                dist2 = (x2 + ck2 - 2.0 * (xb @ ck.t())).squeeze(1)
                dists.append(dist2)
            dist2_full = torch.cat(dists, dim=0)  # [n]
            # select points with labels==k
            mask = labels == k
            if mask.any():
                dist_sel = dist2_full[mask]
                idx_sel_local = torch.nonzero(mask, as_tuple=False).squeeze(1)  # local indices in 0..n-1
                # take top-n closest
                topn = min(rep_per_cluster, dist_sel.numel())
                order = torch.argsort(dist_sel)[:topn]
                chosen_local = idx_sel_local[order].tolist()
                reps_c[k] = [idxs[i] for i in chosen_local]
            else:
                # empty cluster (rare after reinit); just pick a random point
                reps_c[k] = [idxs[int(torch.randint(0, n, (1,)).item())]]

        reps[c] = reps_c

    return reps


def sample_even_from_clusters(cluster_reps: List[List[int]], n: int) -> List[int]:
    """
    Sample n global indices by cycling clusters (even coverage) and sampling within each cluster's reps.
    """
    if not cluster_reps:
        return []

    K = len(cluster_reps)
    out: List[int] = []
    perm = np.random.permutation(K).tolist()
    p = 0

    while len(out) < n:
        if p >= K:
            perm = np.random.permutation(K).tolist()
            p = 0
        k = perm[p]
        p += 1
        bucket = cluster_reps[k]
        if not bucket:
            continue
        out.append(bucket[random.randrange(len(bucket))])

    return out


# ---------------------------
# DREAM runner per dataset
# ---------------------------

def make_args(dataset_name: str) -> SimpleNamespace:
    preset = DATASET_PRESETS[dataset_name]
    d = dict(GLOBAL_PRESET)
    d.update(dict(
        dataset=dataset_name,
        ipc=int(preset["ipc"]),
        Iteration=int(preset["iters"]),
        batch_real=int(preset["batch_real"]),
        batch_train=int(preset["batch_train"]),
        save_path=os.path.join(GLOBAL_PRESET["save_root"], dataset_name),
        # DREAM-specific
        cluster_K=int(preset["cluster_K"]),
        rep_per_cluster=int(preset["rep_per_cluster"]),
    ))
    return SimpleNamespace(**d)


def run_one_dataset(args: SimpleNamespace):
    os.makedirs(args.save_path, exist_ok=True)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dsa_param = ParamDiffAug()
    args.dsa = bool(args.dsa)

    # loops from TARAZ helper
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    # DREAM is more stable if you keep outer_loop moderate
    # (you can override here if you want)
    # args.outer_loop = min(args.outer_loop, 10)

    print("\n============================================================")
    print(f"{get_time()}  DREAM on dataset={args.dataset}")
    print("Preset:", {k: getattr(args, k) for k in [
        "ipc","Iteration","batch_real","batch_train","cluster_K","rep_per_cluster",
        "model","lr_img","lr_net","dis_metric","eval_mode"
    ]})
    print("============================================================")

    # Load dataset via YOUR TARAZ handlers
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    # Build tensors of all training data (like your main_DC.py)
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [int(dst_train[i][1]) for i in range(len(dst_train))]
    # "group"/"bias"/"color" is consistently at index 2 across your datasets
    groups_all = [int(dst_train[i][2]) for i in range(len(dst_train))]

    indices_class: List[List[int]] = [[] for _ in range(num_classes)]
    for i, y in enumerate(labels_all):
        indices_class[y].append(i)

    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all_t = torch.tensor(labels_all, dtype=torch.long, device=args.device)
    groups_all_t = torch.tensor(groups_all, dtype=torch.long, device=args.device)

    args.num_classes = int(len(torch.unique(labels_all_t)))
    args.num_groups = int(len(torch.unique(groups_all_t)))

    for c in range(num_classes):
        print(f"class {c}: {len(indices_class[c])} real images")

    # ---------------------------
    # DREAM: clustering & reps
    # ---------------------------
    print(f"{get_time()}  Building per-class sub-clusters and representative pools...")
    reps = build_representatives_per_class(
        images_all=images_all,
        indices_class=indices_class,
        K=args.cluster_K,
        rep_per_cluster=args.rep_per_cluster,
        feat_hw=8 if min(im_size) >= 32 else 4,
        kmeans_iters=20,
        device=args.device,
    )
    print(f"{get_time()}  Representative pools ready.")

    # Helper to get DREAM-sampled real images
    def get_rep_images(c: int, n: int):
        chosen = sample_even_from_clusters(reps[c], n)
        idx = torch.tensor(chosen, device=args.device, dtype=torch.long)
        return images_all[idx], labels_all_t[idx], groups_all_t[idx]

    # ---------------------------
    # DREAM: clustering-based init
    # ---------------------------
    image_syn = torch.randn(
        size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
        dtype=torch.float,
        requires_grad=True,
        device=args.device,
    )
    label_syn = torch.tensor(
        [np.ones(args.ipc) * i for i in range(num_classes)],
        dtype=torch.long,
        device=args.device,
    ).view(-1)

    color_syn = torch.zeros_like(label_syn)

    print(f"{get_time()}  Initializing synthetic images from sub-cluster representatives...")
    for c in range(num_classes):
        # pick ipc clusters (cycle if ipc > #clusters)
        Kc = len(reps[c])
        if Kc == 0:
            continue
        cluster_order = list(range(Kc))
        random.shuffle(cluster_order)
        chosen_idx = []
        j = 0
        while len(chosen_idx) < args.ipc:
            k = cluster_order[j % Kc]
            j += 1
            bucket = reps[c][k]
            if not bucket:
                continue
            # top-1 rep in the bucket is already "closest to center"
            chosen_idx.append(bucket[0])
        idx = torch.tensor(chosen_idx, device=args.device, dtype=torch.long)

        image_syn.data[c * args.ipc:(c + 1) * args.ipc] = images_all[idx].detach().data
        color_syn.data[c * args.ipc:(c + 1) * args.ipc] = groups_all_t[idx].detach().data

    # Optimizers / loss
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # Evaluate only at the end (preset)
    eval_it_pool = [args.Iteration]

    print(f"{get_time()}  Training begins...")

    # ---------------------------
    # Distillation loop (DC + DREAM sampling)
    # ---------------------------
    for it in range(args.Iteration + 1):

        # ---- Evaluate synthetic data
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print("--------------------------------------------------")
                print(f"{get_time()}  Evaluation: model_train={args.model} model_eval={model_eval} iter={it}")

                if args.dsa:
                    args.epoch_eval_train = 1000
                    args.dc_aug_param = None
                else:
                    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)

                if args.dsa or args.dc_aug_param["strategy"] != "none":
                    args.epoch_eval_train = 1000
                else:
                    args.epoch_eval_train = 300

                accs = []
                max_eo_list, mean_eo_list = [], []
                max_suf_list, mean_suf_list = [], []

                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                    image_syn_eval = copy.deepcopy(image_syn.detach())
                    label_syn_eval = copy.deepcopy(label_syn.detach())
                    _, _, acc_test, max_eo, mean_eo, max_suf, mean_suf = evaluate_synset(
                        it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args
                    )
                    accs.append(acc_test)
                    max_eo_list.append(max_eo)
                    mean_eo_list.append(mean_eo)
                    max_suf_list.append(max_suf)
                    mean_suf_list.append(mean_suf)

                print(f"Acc(mean±std): {np.mean(accs):.4f} ± {np.std(accs):.4f}")
                print(f"EO  (mean): max={np.mean(max_eo_list):.4f}  mean={np.mean(mean_eo_list):.4f}")
                print(f"SUF (mean): max={np.mean(max_suf_list):.4f}  mean={np.mean(mean_suf_list):.4f}")

            # visualize & save
            vis_name = os.path.join(
                args.save_path,
                f"vis_DREAM_{args.dataset}_{args.model}_{args.ipc}ipc_iter{it}.png",
            )
            image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
            for ch in range(channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
            image_syn_vis.clamp_(0.0, 1.0)
            save_image(image_syn_vis, vis_name, nrow=args.ipc)

        # ---- Train synthetic data (one random net per iter, like TARAZ DC)
        net = get_network(args.model, channel, num_classes, im_size).to(args.device)
        net.train()
        net_parameters = list(net.parameters())

        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
        optimizer_net.zero_grad()

        loss_avg = 0.0

        # mute DC aug inside the inner-loop epoch, consistent with your main_DC.py
        args.dc_aug_param = None

        for ol in range(args.outer_loop):

            # Freeze BN running stats using a real batch (fixing the bug in your example)
            BN_flag = any(("BatchNorm" in m._get_name()) for m in net.modules())
            if BN_flag:
                BNSizePC = 16
                imgs_bn = []
                for c in range(num_classes):
                    img_c, _, _ = get_rep_images(c, BNSizePC)
                    imgs_bn.append(img_c)
                img_real_bn = torch.cat(imgs_bn, dim=0)
                _ = net(img_real_bn)
                for m in net.modules():
                    if "BatchNorm" in m._get_name():
                        m.eval()

            # update synthetic images by matching gradients
            loss = torch.tensor(0.0, device=args.device)

            for c in range(num_classes):
                img_real, _, group_real = get_rep_images(c, args.batch_real)
                lab_real = torch.full((img_real.shape[0],), c, device=args.device, dtype=torch.long)

                img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                lab_syn = torch.full((args.ipc,), c, device=args.device, dtype=torch.long)

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                # Default: match to full real batch gradient (DREAM changes *which* real batch we use)
                output_real = net(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = [g.detach().clone() for g in gw_real]

                output_syn = net(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                loss = loss + match_loss(gw_syn, gw_real, args)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += float(loss.item())

            # update network on synthetic set (inner loop)
            image_syn_train = copy.deepcopy(image_syn.detach())
            label_syn_train = copy.deepcopy(label_syn.detach())
            dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
            trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

            for _ in range(args.inner_loop):
                epoch("train", trainloader, net, optimizer_net, criterion, args, aug=True if args.dsa else False)

        loss_avg /= max(1, (num_classes * args.outer_loop))
        # if it % max(1, args.Iteration // 10) == 0 or it == args.Iteration:
        # if it % max(1, args.Iteration // 10) == 0 or it == args.Iteration:
        print(f"{get_time()}  iter={it:04d}  loss={loss_avg:.4f}")

        # save final synthetic set
        if it == args.Iteration:
            data_save = [copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())]
            out_pt = os.path.join(args.save_path, f"res_DREAM_{args.dataset}_{args.model}_{args.ipc}ipc.pt")
            torch.save({"data": data_save}, out_pt)
            print(f"{get_time()}  Saved synthetic set to: {out_pt}")


def main():
    # Fully preset run over your datasets
    set_global_seed(GLOBAL_PRESET["seed"])
    os.makedirs(GLOBAL_PRESET["save_root"], exist_ok=True)

    for dname in DATASETS:
        args = make_args(dname)
        set_global_seed(args.seed)  # keep deterministic per dataset
        run_one_dataset(args)


if __name__ == "__main__":
    main()



# # dream.py
# # DC + DREAM (Representative Matching + Clustering-based Init)
# # Uses TARAZ's data handlers + utils + networks.
# #
# # Drop this file next to main_DC.py and run:
# #   python dream.py
# #
# # Note: DREAM is a plug-in strategy (sampling + init), not a new loss.
# # Paper: "DREAM: Efficient Dataset Distillation by Representative Matching"

# import os
# import time
# import copy
# import random
# from types import SimpleNamespace

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.utils import save_image

# from utils import (
#     get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset,
#     get_daparam, match_loss, get_time, TensorDataset, epoch,
#     DiffAugment, ParamDiffAug
# )

# # ---------------------------
# # Preset experiment config
# # ---------------------------

# DATASETS = (
#     "CIFAR10_S_90",
#     "Colored_FashionMNIST_foreground",
#     "Colored_FashionMNIST_background",
#     "Colored_MNIST_foreground",
#     "Colored_MNIST_background",
#     "UTKface",
# )

# CFG = dict(
#     # Base distillation method (DREAM plugs into this)
#     method="DC",                 # "DC" or "DSA" (DREAM works with both)
#     model="ConvNet",             # training network architecture
#     eval_mode="S",               # evaluation pool mode
#     ipc=10,                      # images per class

#     # Optimization
#     Iteration=1000,
#     lr_img=0.1,
#     lr_net=0.01,
#     batch_real=256,
#     batch_train=256,
#     init="real",                 # DREAM init will override "real" sampling with clustering centers anyway

#     # Augmentation
#     dsa_strategy="None",         # set if using DSA
#     use_dsa=False,               # CFG["method"]=="DSA" normally
#     epoch_eval_train_if_aug=1000,
#     epoch_eval_train_if_noaug=300,

#     # Eval
#     num_exp=1,
#     num_eval=1,

#     # Paths
#     data_path="data",
#     save_path="result_dream",

#     # Matching
#     dis_metric="ours",

#     # FairDD branch (your code)
#     FairDD=False,

#     # DREAM knobs
#     dream_enabled=True,
#     dream_cluster_every=10,       # recompute representative batches every N image-updates
#     dream_kmeans_iters=10,
#     dream_embed_hw=16,            # downsample to (embed_hw, embed_hw) before kmeans

#     # Misc
#     seed=42,
# )


# # ---------------------------
# # Reproducibility helpers
# # ---------------------------

# def set_global_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# # ---------------------------
# # DREAM: embeddings + kmeans
# # ---------------------------

# @torch.no_grad()
# def precompute_embeddings(images_cpu: torch.Tensor, embed_hw: int, device: str, batch_size: int = 2048):
#     """
#     images_cpu: [N,C,H,W] on CPU
#     returns embeddings on CPU: [N, C*embed_hw*embed_hw] float32
#     """
#     N = images_cpu.shape[0]
#     outs = []
#     for s in range(0, N, batch_size):
#         x = images_cpu[s:s+batch_size].to(device, non_blocking=True)
#         x = F.adaptive_avg_pool2d(x, (embed_hw, embed_hw))
#         x = x.flatten(1)
#         x = F.normalize(x, dim=1)
#         outs.append(x.detach().cpu().float())
#     return torch.cat(outs, dim=0)  # CPU float32


# def kmeans_representatives(X: torch.Tensor, k: int, iters: int, seed: int):
#     """
#     X: [N,D] on GPU/CPU float32
#     returns:
#       rep_local_idx: list[int] length k (indices in 0..N-1)
#     """
#     N, D = X.shape
#     if k <= 0:
#         return []
#     if N <= k:
#         return list(range(N))

#     gen = torch.Generator(device=X.device)
#     gen.manual_seed(seed)

#     # init centers
#     init_idx = torch.randperm(N, generator=gen, device=X.device)[:k]
#     centers = X[init_idx].clone()

#     for _ in range(iters):
#         # assignment
#         dist = torch.cdist(X, centers)  # [N,k]
#         assign = dist.argmin(dim=1)     # [N]

#         # update centers using scatter_add
#         counts = torch.bincount(assign, minlength=k).float().unsqueeze(1)  # [k,1]
#         sums = torch.zeros((k, D), device=X.device, dtype=X.dtype)
#         sums.scatter_add_(0, assign.unsqueeze(1).expand(-1, D), X)

#         new_centers = sums / counts.clamp_min(1.0)

#         # keep old center for empty clusters
#         empty = (counts.squeeze(1) == 0)
#         if empty.any():
#             new_centers[empty] = centers[empty]
#         centers = new_centers

#     # pick nearest sample to each center
#     dist = torch.cdist(X, centers)  # [N,k]
#     assign = dist.argmin(dim=1)

#     reps = []
#     for j in range(k):
#         idxs = torch.nonzero(assign == j, as_tuple=False).squeeze(1)
#         if idxs.numel() == 0:
#             # fallback random
#             reps.append(int(torch.randint(0, N, (1,), generator=gen, device=X.device).item()))
#             continue
#         dj = dist[idxs, j]
#         reps.append(int(idxs[dj.argmin()].item()))
#     return reps


# def build_rep_idx_per_class(emb_cpu: torch.Tensor, indices_class: list, k_per_class: int,
#                             iters: int, seed: int, device: str):
#     """
#     emb_cpu: [N,D] on CPU
#     indices_class: list[list[int]] where indices_class[c] are global indices of class c
#     returns dict c -> np.array of global indices (representatives)
#     """
#     rep = {}
#     for c, idxs in enumerate(indices_class):
#         idxs = list(idxs)
#         if len(idxs) == 0:
#             rep[c] = np.array([], dtype=np.int64)
#             continue

#         k = min(k_per_class, len(idxs))
#         X = emb_cpu[idxs].to(device)
#         reps_local = kmeans_representatives(X, k=k, iters=iters, seed=seed + 1000 * c)
#         reps_global = np.array([idxs[i] for i in reps_local], dtype=np.int64)
#         rep[c] = reps_global
#     return rep


# def sample_from_rep(images_cpu, labels_cpu, groups_cpu, rep_idx: np.ndarray, n: int):
#     """
#     Return a batch of size n sampled from representative indices (with replacement if needed).
#     """
#     if rep_idx.size == 0:
#         return None

#     if rep_idx.size >= n:
#         chosen = np.random.choice(rep_idx, size=n, replace=False)
#     else:
#         chosen = np.random.choice(rep_idx, size=n, replace=True)

#     img = images_cpu[chosen]
#     lab = labels_cpu[chosen]
#     grp = groups_cpu[chosen]
#     return img, lab, grp


# # ---------------------------
# # Main distillation for 1 dataset
# # ---------------------------

# def run_one_dataset(dataset_name: str, cfg: dict):
#     args = SimpleNamespace(**cfg)
#     args.dataset = dataset_name
#     args.dsa_param = ParamDiffAug()
#     args.dsa = bool(args.use_dsa) or (args.method.upper() == "DSA")

#     # device
#     args.device = "cuda" if torch.cuda.is_available() else "cpu"

#     # output dir
#     os.makedirs(args.save_path, exist_ok=True)

#     # loops (baseline DC heuristic)
#     args.outer_loop, args.inner_loop = get_loops(args.ipc)

#     # eval checkpoints
#     eval_it_pool = [args.Iteration]

#     print(f"\n================== DATASET: {args.dataset} ==================\n")
#     channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
#     model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

#     # ---------------------------
#     # Load all training data into CPU tensors (your datasets here are manageable)
#     # ---------------------------
#     images_all = []
#     labels_all = []
#     groups_all = []

#     for i in range(len(dst_train)):
#         datum = dst_train[i]
#         images_all.append(datum[0].unsqueeze(0).cpu())
#         labels_all.append(int(datum[1]))
#         groups_all.append(int(datum[2]))

#     images_all = torch.cat(images_all, dim=0)                      # CPU [N,C,H,W]
#     labels_all = torch.tensor(labels_all, dtype=torch.long).cpu()  # CPU [N]
#     groups_all = torch.tensor(groups_all, dtype=torch.long).cpu()  # CPU [N]

#     # Update args for fairness metric code
#     args.num_classes = int(labels_all.unique().numel())
#     args.num_groups = int(groups_all.unique().numel())

#     # Indices per class
#     indices_class = [[] for _ in range(num_classes)]
#     for i, lab in enumerate(labels_all.tolist()):
#         indices_class[lab].append(i)

#     for c in range(num_classes):
#         print(f"class {c}: {len(indices_class[c])} real images")

#     # ---------------------------
#     # DREAM: precompute embeddings (for KMeans)
#     # ---------------------------
#     emb_cpu = None
#     if args.dream_enabled:
#         print(f"{get_time()} DREAM: precomputing embeddings (adaptive pool to {args.dream_embed_hw}x{args.dream_embed_hw}) ...")
#         emb_cpu = precompute_embeddings(images_all, embed_hw=args.dream_embed_hw, device=args.device)

#     # ---------------------------
#     # Initialize synthetic set
#     # ---------------------------
#     image_syn = torch.randn(
#         size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
#         dtype=torch.float, requires_grad=True, device=args.device
#     )
#     label_syn = torch.tensor(
#         [np.ones(args.ipc) * i for i in range(num_classes)],
#         dtype=torch.long, device=args.device
#     ).view(-1)

#     # DREAM init: cluster each class into ipc clusters, pick center samples
#     if args.init == "real":
#         if args.dream_enabled and emb_cpu is not None:
#             print(f"{get_time()} DREAM init: clustering each class into ipc={args.ipc} reps for initialization ...")
#             rep_init = build_rep_idx_per_class(
#                 emb_cpu=emb_cpu,
#                 indices_class=indices_class,
#                 k_per_class=args.ipc,
#                 iters=args.dream_kmeans_iters,
#                 seed=cfg["seed"],
#                 device=args.device
#             )
#             for c in range(num_classes):
#                 reps = rep_init[c]
#                 if reps.size == 0:
#                     continue
#                 # if class has fewer than ipc samples, oversample reps
#                 if reps.size < args.ipc:
#                     reps = np.random.choice(reps, size=args.ipc, replace=True)
#                 img0 = images_all[reps].to(args.device)
#                 image_syn.data[c * args.ipc:(c + 1) * args.ipc] = img0.detach()
#         else:
#             print("init=real (random real images per class)")
#             for c in range(num_classes):
#                 idxs = np.random.permutation(indices_class[c])[:args.ipc]
#                 image_syn.data[c * args.ipc:(c + 1) * args.ipc] = images_all[idxs].to(args.device).detach()
#     else:
#         print("init=noise")

#     # ---------------------------
#     # Optimizers
#     # ---------------------------
#     optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
#     criterion = nn.CrossEntropyLoss().to(args.device)

#     # For logging
#     accs_all_exps = {m: [] for m in model_eval_pool}

#     # ---------------------------
#     # DREAM representative indices (for matching)
#     # ---------------------------
#     rep_match = None
#     img_update_step = 0

#     def get_real_batch_for_class(c: int, n: int):
#         nonlocal rep_match
#         if args.dream_enabled and (rep_match is not None) and (c in rep_match) and rep_match[c].size > 0:
#             out = sample_from_rep(images_all, labels_all, groups_all, rep_match[c], n)
#             if out is not None:
#                 img, lab, grp = out
#                 return img.to(args.device), lab.to(args.device), grp.to(args.device)

#         # fallback random sampling
#         idxs = np.random.permutation(indices_class[c])[:min(n, len(indices_class[c]))]
#         if len(idxs) < n:
#             idxs = np.random.choice(indices_class[c], size=n, replace=True)
#         img = images_all[idxs].to(args.device)
#         lab = labels_all[idxs].to(args.device)
#         grp = groups_all[idxs].to(args.device)
#         return img, lab, grp

#     print(f"{get_time()} training begins")

#     # ---------------------------
#     # Training loop
#     # ---------------------------
#     for it in range(args.Iteration + 1):

#         # Evaluate
#         if it in eval_it_pool:
#             for model_eval in model_eval_pool:
#                 print(f"-------------------------\nEvaluation\ntrain={args.model}, eval={model_eval}, it={it}")
#                 if args.dsa:
#                     args.dc_aug_param = None
#                     args.epoch_eval_train = args.epoch_eval_train_if_aug
#                 else:
#                     args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)
#                     if args.dc_aug_param["strategy"] != "none":
#                         args.epoch_eval_train = args.epoch_eval_train_if_aug
#                     else:
#                         args.epoch_eval_train = args.epoch_eval_train_if_noaug

#                 accs = []
#                 max_eo_list, mean_eo_list = [], []
#                 for it_eval in range(args.num_eval):
#                     net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
#                     image_syn_eval = copy.deepcopy(image_syn.detach())
#                     label_syn_eval = copy.deepcopy(label_syn.detach())
#                     _, acc_train, acc_test, max_eo, mean_eo, max_suff, mean_suff = evaluate_synset(
#                         it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args
#                     )
#                     accs.append(acc_test)
#                     max_eo_list.append(max_eo)
#                     mean_eo_list.append(mean_eo)

#                 print(f"Evaluate {len(accs)} random {model_eval}, mean={np.mean(accs):.4f} std={np.std(accs):.4f}")
#                 print("acc, max_EO, mean_EO:", np.mean(accs), np.mean(max_eo_list), np.mean(mean_eo_list))

#                 if it == args.Iteration:
#                     accs_all_exps[model_eval] += accs

#             # visualize
#             save_name = os.path.join(
#                 args.save_path,
#                 f"vis_DREAM_{args.dataset}_{args.model}_{args.ipc}ipc_iter{it}.png"
#             )
#             image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
#             for ch in range(channel):
#                 image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
#             image_syn_vis.clamp_(0.0, 1.0)
#             save_image(image_syn_vis, save_name, nrow=args.ipc)

#         # Create random net
#         net = get_network(args.model, channel, num_classes, im_size).to(args.device)
#         net.train()
#         net_parameters = list(net.parameters())
#         optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
#         criterion = nn.CrossEntropyLoss().to(args.device)

#         # mute DC aug in inner loop for consistency
#         args.dc_aug_param = None

#         loss_avg = 0.0

#         for ol in range(args.outer_loop):

#             # DREAM: recompute representative real indices periodically
#             if args.dream_enabled and (emb_cpu is not None):
#                 if (img_update_step % args.dream_cluster_every) == 0:
#                     # pick batch_real reps per class
#                     rep_match = build_rep_idx_per_class(
#                         emb_cpu=emb_cpu,
#                         indices_class=indices_class,
#                         k_per_class=args.batch_real,
#                         iters=args.dream_kmeans_iters,
#                         seed=cfg["seed"] + img_update_step,
#                         device=args.device
#                     )
#                 img_update_step += 1

#             # BatchNorm handling (fixed from your main_DC.py bug)
#             BN_flag = any("BatchNorm" in m._get_name() for m in net.modules())
#             if BN_flag:
#                 BNSizePC = 16
#                 imgs_bn = []
#                 for c in range(num_classes):
#                     img_c, _, _ = get_real_batch_for_class(c, BNSizePC)
#                     imgs_bn.append(img_c)
#                 img_real_bn = torch.cat(imgs_bn, dim=0)
#                 net.train()
#                 _ = net(img_real_bn)
#                 for module in net.modules():
#                     if "BatchNorm" in module._get_name():
#                         module.eval()

#             # Update synthetic data by matching gradients
#             loss = torch.tensor(0.0, device=args.device)

#             for c in range(num_classes):
#                 img_real, _, grp_real = get_real_batch_for_class(c, args.batch_real)
#                 lab_real = torch.full((img_real.shape[0],), c, device=args.device, dtype=torch.long)

#                 img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(args.ipc, channel, im_size[0], im_size[1])
#                 lab_syn = torch.full((args.ipc,), c, device=args.device, dtype=torch.long)

#                 if args.dsa:
#                     seed_aug = int(time.time() * 1000) % 100000
#                     img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed_aug, param=args.dsa_param)
#                     img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed_aug, param=args.dsa_param)

#                 output_real = net(img_real)
#                 output_syn = net(img_syn)

#                 loss_syn = criterion(output_syn, lab_syn)
#                 gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

#                 if args.FairDD:
#                     # match each group gradient to synthetic gradient
#                     for g in torch.unique(grp_real):
#                         mask = (grp_real == g)
#                         if mask.sum() == 0:
#                             continue
#                         loss_real_g = criterion(output_real[mask], lab_real[mask])
#                         gw_real_g = torch.autograd.grad(loss_real_g, net_parameters, retain_graph=True)
#                         gw_real_g = [_.detach().clone() for _ in gw_real_g]
#                         loss = loss + match_loss(gw_syn, gw_real_g, args)
#                 else:
#                     loss_real = criterion(output_real, lab_real)
#                     gw_real = torch.autograd.grad(loss_real, net_parameters)
#                     gw_real = [_.detach().clone() for _ in gw_real]
#                     loss = loss + match_loss(gw_syn, gw_real, args)

#             optimizer_img.zero_grad()
#             loss.backward()
#             optimizer_img.step()
#             loss_avg += float(loss.item())

#             # Update net (inner loop)
#             image_syn_train = copy.deepcopy(image_syn.detach())
#             label_syn_train = copy.deepcopy(label_syn.detach())
#             dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
#             trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
#             optimizer_net.zero_grad()
#             for _ in range(args.inner_loop):
#                 epoch("train", trainloader, net, optimizer_net, criterion, args, aug=True if args.dsa else False)

#         loss_avg /= max(1, (num_classes * args.outer_loop))
#         print(f"{get_time()} it={it:04d}, loss={loss_avg:.4f}")

#         # Save final
#         if it == args.Iteration:
#             data_save = [copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())]
#             out_pt = os.path.join(args.save_path, f"res_DREAM_{args.dataset}_{args.model}_{args.ipc}ipc.pt")
#             torch.save({"data": data_save, "accs_all_exps": accs_all_exps}, out_pt)
#             print(f"Saved synthetic data to: {out_pt}")

#     print("\n==================== Final Results ====================\n")
#     for key in model_eval_pool:
#         accs = accs_all_exps[key]
#         if len(accs) == 0:
#             continue
#         print(f"Dataset {args.dataset} | train {args.model} | eval {key} | mean={np.mean(accs):.2f}% std={np.std(accs):.2f}%")


# def main():
#     set_global_seed(CFG["seed"])
#     os.makedirs(CFG["save_path"], exist_ok=True)

#     # If you want DSA+DREAM, flip these:
#     # CFG["method"]="DSA"; CFG["use_dsa"]=True; CFG["dsa_strategy"]="color_crop_cutout_flip_scale_rotate" (example)

#     for d in DATASETS:
#         try:
#             run_one_dataset(d, CFG)
#         except Exception as e:
#             print(f"\n[ERROR] Dataset {d} failed: {e}\n")


# if __name__ == "__main__":
#     main()
