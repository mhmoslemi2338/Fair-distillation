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


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="DC")
    parser.add_argument("--dataset", type=str, default="Colored_FashionMNIST_background")
    parser.add_argument("--model", type=str, default="ConvNet")
    parser.add_argument("--ipc", type=int, default=50)
    parser.add_argument("--eval_mode", type=str, default="S")
    parser.add_argument("--num_eval", type=int, default=20)

    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--results_root", type=str, default="/home/mmoslem3/scratch/FairDD/results")
    parser.add_argument("--out_dir", type=str, default="/home/mmoslem3/scratch/FairDD/dispersion-plots")

    parser.add_argument("--testMetric", type=str, default="DC")
    parser.add_argument("--param_mode", type=str, default="all", choices=["last_linear", "all"])
    parser.add_argument("--normalize", type=str, default="by_all_norm", choices=["none", "by_all_norm"])

    # used by your codebase
    parser.add_argument("--batch_train", type=int, default=256)
    parser.add_argument("--batch_real", type=int, default=256)
    parser.add_argument("--dsa_strategy", type=str, default="None")
    parser.add_argument("--lr_net", type=float, default=0.01)
    parser.add_argument("--epoch_eval_train", type=int, default=300)
    parser.add_argument("--FairDD", action="store_true")

    args = parser.parse_args()


    
    for dataset in [
                    "CIFAR10_S_90",
                    "Colored_FashionMNIST_foreground",
                    # "Colored_FashionMNIST_background",
                    "Colored_MNIST_foreground",
                    "Colored_MNIST_background",
                    "UTKface"
                    ]:
        args.dataset = dataset
        # for ipc in [10, 50, 100]:
        for ipc in [10, 50]:
            args.ipc = ipc


            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required (your fairness metric code uses .cuda()).")
            args.device = "cuda"

            load_random_state(random_state)

            args.outer_loop, args.inner_loop = get_loops(args.ipc)
            args.dsa_param = ParamDiffAug()
            args.dsa = True if args.method == "DSA" else False

            channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(
                args.dataset, args.data_path
            )
            model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
            model_eval = model_eval_pool[0]

            # full real tensors (train set)
            images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
            labels_all = [dst_train[i][1] for i in range(len(dst_train))]
            images_all = torch.cat(images_all, dim=0).to(args.device)
            labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

            # needed by utils fairness evaluation
            args.num_classes = int(num_classes)
            if len(dst_train) > 0 and isinstance(dst_train[0], (tuple, list)) and len(dst_train[0]) >= 3:
                color_all = torch.tensor([dst_train[i][2] for i in range(len(dst_train))], dtype=torch.long, device=args.device)
                args.num_groups = int(len(torch.unique(color_all)))
            else:
                args.num_groups = 1

            args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)
            if args.dsa or args.dc_aug_param.get("strategy", "none") != "none":
                args.epoch_eval_train = max(args.epoch_eval_train, 1000)


            # --- choose how many full-training runs you want (fixed) ---
            num_full = 4  # train Full only this many times

            order = ("Full", "NoFair", "FairDD", "NoOrtho")
            per_eval_dists = {k: [] for k in order}
            norms = {k: [] for k in order}
            gids_ref = None

            # --------------------------
            # 1) FULL: only num_full runs
            # --------------------------
            for it_full in range(num_full):
                net0 = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                init_state = copy.deepcopy(net0.state_dict())

                net_full = train_from_init_state(
                    it_full, model_eval, channel, num_classes, im_size, args.device,
                    init_state, images_all, labels_all, testloader, args
                )

                gids_cur, d_cur, all_norm = per_group_dispersion_distances(
                    net_full, testloader, args.device,
                    param_mode=args.param_mode, normalize=args.normalize
                )
                norms["Full"].append(all_norm)

                if gids_ref is None:
                    gids_ref = gids_cur

                d_aligned = align_to_reference(gids_ref, gids_cur, d_cur)
                per_eval_dists["Full"].append(d_aligned)

            # --------------------------
            # 2) SYNTH: args.num_eval runs (independent from Full)
            # --------------------------
            for it_eval in range(args.num_eval):
                net0 = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                init_state = copy.deepcopy(net0.state_dict())

                # Train each synthetic method from the SAME init_state for fairness of comparison
                trained = {}

                for fair_crt in ("NoFair", "FairDD", "NoOrtho"):
                    ckpt, path_used = load_synthetic_checkpoint(
                        args.results_root, args.testMetric, fair_crt, args.dataset, args.ipc, args.device
                    )
                    image_syn, label_syn = extract_syn_data(ckpt, args.device)

                    net_syn = train_from_init_state(
                        it_eval, model_eval, channel, num_classes, im_size, args.device,
                        init_state, image_syn.detach().clone(), label_syn.detach().clone(), testloader, args
                    )
                    trained[fair_crt] = net_syn
                    print(f"Loaded {fair_crt} from: {path_used}")

                # Compute distances for each synthetic method
                for k in ("NoFair", "FairDD", "NoOrtho"):
                    gids_cur, d_cur, all_norm = per_group_dispersion_distances(
                        trained[k], testloader, args.device,
                        param_mode=args.param_mode, normalize=args.normalize
                    )
                    norms[k].append(all_norm)

                    if gids_ref is None:
                        gids_ref = gids_cur

                    d_aligned = align_to_reference(gids_ref, gids_cur, d_cur)
                    per_eval_dists[k].append(d_aligned)

            # --------------------------
            # 3) Aggregate: Full uses num_full, others use args.num_eval
            # --------------------------
            dists_mean = {}
            dists_std = {}

            for k in order:
                arr = np.stack(per_eval_dists[k], axis=0)  # [n_runs_for_k, num_groups]
                dists_mean[k] = np.nanmean(arr, axis=0)
                # std only meaningful if >= 2 runs
                if arr.shape[0] >= 2:
                    dists_std[k] = np.nanstd(arr, axis=0, ddof=0)
                else:
                    dists_std[k] = None





            out_dir = os.path.join(args.out_dir, f"{args.testMetric}-{args.dataset}-ipc{args.ipc}")

            # Plot 1: x=groups, 4 bars per group
            plot_groups_x_methods_bars(
                dists_mean=dists_mean,
                dists_std=dists_std,
                gids=gids_ref if gids_ref is not None else [0],
                dataset=args.dataset,
                ipc=args.ipc,
                out_dir=out_dir,
                order=order,
                param_mode=args.param_mode,
                normalize=args.normalize,
            )

            # Plot 2: x=methods, bars are groups
            plot_methods_x_groups_bars(
                dists_mean=dists_mean,
                dists_std=dists_std,
                gids=gids_ref if gids_ref is not None else [0],
                dataset=args.dataset,
                ipc=args.ipc,
                out_dir=out_dir,
                order=order,
                param_mode=args.param_mode,
                normalize=args.normalize,
            )

            # useful if normalize=by_all_norm
            for k in order:
                if len(norms[k]) > 0:
                    print(f"{k}: mean ||g_all|| = {float(np.mean(norms[k])):.6f}")


            save_dir = out_dir
            os.makedirs(save_dir, exist_ok=True)

            save_payload = {
                "meta": {
                    "dataset": args.dataset,
                    "ipc": args.ipc,
                    "testMetric": args.testMetric,
                    "param_mode": args.param_mode,
                    "normalize": args.normalize,
                    "num_eval": args.num_eval,
                    "method": args.method,
                    "model": args.model,
                    "eval_mode": args.eval_mode,
                },
                "order": tuple(order),
                "gids_ref": list(map(int, gids_ref if gids_ref is not None else [])),
                "dists_mean": {k: np.asarray(v, dtype=np.float64) for k, v in dists_mean.items()},
                "dists_std": None if dists_std is None else {k: np.asarray(v, dtype=np.float64) for k, v in dists_std.items()},
                # optional but recommended: raw per-eval aligned distances and grad norms
                "per_eval_dists": {k: np.stack(per_eval_dists[k], axis=0) for k in order},  # [num_eval, num_groups]
                "g_all_norms": {k: np.asarray(norms[k], dtype=np.float64) for k in order},  # length=num_eval
            }

            pt_path = os.path.join(save_dir, f"dispersion_metrics_{args.dataset}_ipc{args.ipc}_{args.param_mode}_{args.normalize}.pt")
            torch.save(save_payload, pt_path)
            print(f"Saved metrics to: {pt_path}")


            # ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            # gids_ref = ckpt["gids_ref"]
            # order = ckpt["order"]
            # dists_mean = ckpt["dists_mean"]
            # dists_std = ckpt["dists_std"]


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

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random_state = save_random_state()
    main()
