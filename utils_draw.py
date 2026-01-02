

#!/usr/bin/env python3
import os
import copy
import argparse
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    get_loops,
    get_dataset,
    get_network,
    get_eval_pool,
    evaluate_synset,
    get_daparam,
    ParamDiffAug,
)

# ----------------------------
# Checkpoints
# ----------------------------
def build_checkpoint_candidates(results_root, name, fair_crt, dataset, ipc):
    base = os.path.join(results_root, f"{name}-{fair_crt}")

    if fair_crt == "FairDD":
        dir1 = os.path.join(base, f"FairDD_{name}_{dataset}_ipc{ipc}")
        file1 = os.path.join(dir1, f"res_{name}_{dataset}_ConvNet_{ipc}ip1000.pt")
        return [file1, file1.replace("1000", "700"), file1.replace("1000", "600"), file1.replace("1000", "550")]

    if fair_crt == "NoOrtho":
        dir1 = os.path.join(base, f"Fair_NoOrtho_{name}_{dataset}_ipc{ipc}")
        file1 = os.path.join(dir1, f"res_{name}_{dataset}_ConvNet_{ipc}ip1000.pt")
        return [file1, file1.replace("1000", "700"), file1.replace("1000", "600"), file1.replace("1000", "550")]

    if fair_crt == "NoFair":
        dir1 = os.path.join(base, f"{name}_{dataset}_ipc{ipc}")
        file1 = os.path.join(dir1, f"res_{name}_{dataset}_ConvNet_{ipc}ipc.pt")
        return [file1]

    raise ValueError(f"Unknown fair_crt: {fair_crt}")


def load_synthetic_checkpoint(results_root, name, fair_crt, dataset, ipc, device):
    candidates = build_checkpoint_candidates(results_root, name, fair_crt, dataset, ipc)
    last_err = None
    for p in candidates:
        try:
            ckpt = torch.load(p, map_location=device, weights_only=False)
            return ckpt, p
        except Exception as e:
            last_err = e
    raise FileNotFoundError(
        f"No checkpoint found for {fair_crt} using candidates:\n  "
        + "\n  ".join(candidates)
        + f"\nLast error: {last_err}"
    )


def extract_syn_data(checkpoint, device):
    try:
        image_syn, label_syn = checkpoint["data"][0]
    except Exception:
        image_syn, label_syn = checkpoint["data"]
    return image_syn.to(device), label_syn.to(device)


# ----------------------------
# Training: same init for all methods
# ----------------------------
def train_from_init_state(it_eval, model_eval, channel, num_classes, im_size, device,
                          init_state, train_images, train_labels, testloader, args):
    net = get_network(model_eval, channel, num_classes, im_size).to(device)
    net.load_state_dict(copy.deepcopy(init_state), strict=True)
    net_trained, *_ = evaluate_synset(it_eval, net, train_images, train_labels, testloader, args, verbose=False)
    return net_trained


# ----------------------------
# Grad utilities
# ----------------------------
def pick_last_linear_params(model):
    named = [(n, p) for (n, p) in model.named_parameters() if p.requires_grad]
    twod = [(n, p) for (n, p) in named if p.ndim == 2]
    if len(twod) == 0:
        return [p for _, p in named]
    last_name, last_W = twod[-1]
    params = [last_W]
    bias_name = last_name.replace("weight", "bias")
    for n, p in named:
        if n == bias_name:
            params.append(p)
            break
    return params


def pick_all_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def grads_to_vec(grads_list):
    return torch.cat([g.detach().flatten() for g in grads_list])


def unpack_batch(batch, device):
    # Supports (x, y) or (x, y, a, ...)
    x = batch[0].to(device)
    y = batch[1].to(device)
    if len(batch) >= 3:
        a = batch[2].long().to(device)
    else:
        a = torch.zeros_like(y, dtype=torch.long, device=device)
    return x, y, a


def grad_vec_all_and_per_group(model, loader, device, params):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    sum_all = [torch.zeros_like(p, device=device) for p in params]
    n_all = 0

    sum_g = {}  # gid -> list[tensor]
    n_g = {}    # gid -> count

    for batch in loader:
        x, y, a = unpack_batch(batch, device)

        # full batch grad
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        g = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)

        bs = int(y.numel())
        n_all += bs
        for i in range(len(params)):
            sum_all[i] += g[i].detach()

        # per-group grads: recompute on subset
        uniq_gids = torch.unique(a).tolist()
        for gid in uniq_gids:
            gid = int(gid)
            mask = (a == gid)
            k = int(mask.sum().item())
            if k == 0:
                continue

            if gid not in sum_g:
                sum_g[gid] = [torch.zeros_like(p, device=device) for p in params]
                n_g[gid] = 0

            x_g = x[mask]
            y_g = y[mask]

            model.zero_grad(set_to_none=True)
            out_g = model(x_g)
            loss_g = criterion(out_g, y_g)
            g_g = torch.autograd.grad(loss_g, params, retain_graph=False, create_graph=False)

            n_g[gid] += k
            for i in range(len(params)):
                sum_g[gid][i] += g_g[i].detach()

    mean_all = [gg / max(n_all, 1) for gg in sum_all]
    v_all = grads_to_vec(mean_all)

    v_g = {}
    for gid in sum_g.keys():
        denom = max(n_g[gid], 1)
        mean_gid = [gg / denom for gg in sum_g[gid]]
        v_g[gid] = grads_to_vec(mean_gid)

    return v_all, v_g, n_g


def per_group_dispersion_distances(model, loader, device, param_mode="last_linear", normalize="by_all_norm"):
    if param_mode == "all":
        params = pick_all_trainable_params(model)
    else:
        params = pick_last_linear_params(model)

    v_all, v_g, _ = grad_vec_all_and_per_group(model, loader, device, params)

    eps = 1e-12
    all_norm = torch.norm(v_all).item()

    gids_sorted = sorted(v_g.keys())
    d = []
    for gid in gids_sorted:
        dist = torch.norm(v_all - v_g[gid]).item()
        if normalize == "by_all_norm":
            dist = dist / (all_norm + eps)
        d.append(dist)

    return gids_sorted, np.array(d, dtype=np.float64), all_norm


def align_to_reference(gids_ref, gids_cur, d_cur):
    m = {int(g): float(v) for g, v in zip(gids_cur, d_cur)}
    return np.array([m.get(int(g), np.nan) for g in gids_ref], dtype=np.float64)


# ----------------------------
# Plot 1: x = groups, 4 bars per group (methods)
# ----------------------------
def plot_groups_x_methods_bars(dists_mean, dists_std, gids, dataset, ipc, out_dir,
                               order=("Full", "NoFair", "FairDD", "NoOrtho"),
                               param_mode="last_linear", normalize="by_all_norm"):
    os.makedirs(out_dir, exist_ok=True)

    x = np.arange(len(gids))
    width = 0.20
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(order))

    plt.figure(figsize=(14, 5))
    for i, k in enumerate(order):
        y = dists_mean[k]
        yerr = dists_std[k] if dists_std is not None else None
        plt.bar(x + offsets[i], y, width, label=k, yerr=yerr, capsize=3 if yerr is not None else 0)

    plt.xticks(x, [str(g) for g in gids])
    plt.xlabel("Group (sensitive attribute)")
    ylabel = "||g_all - g_group||_2"
    if normalize == "by_all_norm":
        ylabel += " / (||g_all||_2)"
    plt.ylabel(ylabel)

    title = f"{dataset} | ipc={ipc} | per-group dispersion"
    title += " | params=last_linear" if param_mode == "last_linear" else " | params=all"
    plt.title(title)

    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    fn = os.path.join(out_dir, f"plot1_groups_x_methods_{dataset}_ipc{ipc}_{param_mode}_{normalize}.png")
    plt.tight_layout()
    plt.savefig(fn, dpi=200)
    plt.close()
    print(f"Saved: {fn}")


# ----------------------------
# Plot 2: x = methods, bars are groups within each method
# ----------------------------
def plot_methods_x_groups_bars(dists_mean, dists_std, gids, dataset, ipc, out_dir,
                               order=("Full", "NoFair", "FairDD", "NoOrtho"),
                               param_mode="last_linear", normalize="by_all_norm"):
    os.makedirs(out_dir, exist_ok=True)

    x = np.arange(len(order))
    G = len(gids)
    width = min(0.80 / max(G, 1), 0.25)  # keep within reasonable width
    offsets = (np.arange(G) - (G - 1) / 2.0) * width

    plt.figure(figsize=(12, 5))

    for gi, gid in enumerate(gids):
        y = np.array([dists_mean[m][gi] for m in order], dtype=np.float64)
        if dists_std is not None:
            yerr = np.array([dists_std[m][gi] for m in order], dtype=np.float64)
        else:
            yerr = None
        plt.bar(x + offsets[gi], y, width, label=f"Group {gid}", yerr=yerr, capsize=3 if yerr is not None else 0)

    plt.xticks(x, list(order))
    plt.xlabel("Method")
    ylabel = "||g_all - g_group||_2"
    if normalize == "by_all_norm":
        ylabel += " / (||g_all||_2)"
    plt.ylabel(ylabel)

    title = f"{dataset} | ipc={ipc} | methods vs groups"
    title += " | params=last_linear" if param_mode == "last_linear" else " | params=all"
    plt.title(title)

    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    fn = os.path.join(out_dir, f"plot2_methods_x_groups_{dataset}_ipc{ipc}_{param_mode}_{normalize}.png")
    plt.tight_layout()
    plt.savefig(fn, dpi=200)
    plt.close()
    print(f"Saved: {fn}")

