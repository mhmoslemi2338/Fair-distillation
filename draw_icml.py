import os
import numpy as np
import torch

# ---- set these in your notebook ----
metrics_root = "dispersion-plots"
testMetric   = "DC"
param_mode   = "all"         # or "last_linear"
normalize    = "by_all_norm" # or "none"



# -----------------------------------

def safe_load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")

def to_np(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# container for everything you’ll plot yourself
runs = {}  # key: (dataset, ipc) -> dict of arrays


dataset =[
    "CIFAR10_S_90",
    "Colored_FashionMNIST_foreground",
    "Colored_FashionMNIST_background",
    "Colored_MNIST_foreground",
    "Colored_MNIST_background",
    "UTKface"]


for ds in dataset:
    # for ipc in :
    for i in [10,50,100]:
        case_dir = os.path.join(metrics_root, f"{testMetric}-{ds}-ipc{i}")
        pt_path  = os.path.join(
            case_dir, f"dispersion_metrics_{ds}_ipc{i}_{param_mode}_{normalize}.pt"
        )

        if not os.path.isfile(pt_path):
            print(f"missing: {pt_path}")
            continue

        ckpt = safe_load_pt(pt_path)

        order = tuple(ckpt.get("order", ("Full", "NoFair", "FairDD", "NoOrtho")))
        gids_ref = np.array(ckpt.get("gids_ref", []), dtype=int)

        dists_mean = {k: to_np(ckpt["dists_mean"][k]) for k in order}
        dists_std_raw = ckpt.get("dists_std", None)
        dists_std = (
            {k: (to_np(dists_std_raw.get(k)) if dists_std_raw.get(k) is not None else None) for k in order}
            if isinstance(dists_std_raw, dict) else {k: None for k in order}
        )

        per_eval_dists_raw = ckpt.get("per_eval_dists", None)
        per_eval_dists = (
            {k: to_np(per_eval_dists_raw[k]) for k in order}
            if isinstance(per_eval_dists_raw, dict) else None
        )

        g_all_norms_raw = ckpt.get("g_all_norms", None)
        g_all_norms = (
            {k: to_np(g_all_norms_raw.get(k, [])) for k in order}
            if isinstance(g_all_norms_raw, dict) else None
        )

        runs[(ds, i)] = dict(
            pt_path=pt_path,
            meta=ckpt.get("meta", {}),
            order=order,
            gids_ref=gids_ref,
            dists_mean=dists_mean,
            dists_std=dists_std,
            per_eval_dists=per_eval_dists,
            g_all_norms=g_all_norms,
        )

# now you can do e.g.
# r = runs[(dataset, ipc)]
# r["dists_mean"]["FairDD"]  # -> numpy array [num_groups]
# r["per_eval_dists"]["FairDD"]  # -> numpy array [num_runs, num_groups] (if saved)



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---- color-blind friendly base colors for the 4 METHODS (ICML-safe) ----
# Full Dataset (neutral gray), DC (orange), DC+FairDD (blue), DC+TARAZ (green)
base_hex = ["#4D4D4D", "#E69F00", "#0072B2", "#009E73"]
base_rgb = [np.array(mcolors.to_rgb(c)) for c in base_hex]

def lighten(rgb, amt):
    """Blend rgb toward white by amt in [0,1]. 0=no change, 1=white."""
    rgb = np.array(rgb)
    return tuple((1 - amt) * rgb + amt * np.ones(3))

max_light = 0.70      # how light the lightest subgroup gets
dev_fontsize = 7      # text size for |bar-mean| labels (can get busy if many groups)
dev_alpha = 0.55      # alpha for deviation text

# ----------------------------------------------------------------------

for ds in dataset[1:]:
    for ipc_val in [10,50,100]:

        r = runs[(ds, ipc_val)]

        order = list(r["order"])  # expected 4 methods
        means = np.stack([np.asarray(r["dists_mean"][m]) for m in order], axis=0)  # [M,G]
        M, G = means.shape

        # optional stds if you want later
        stds = None
        if isinstance(r.get("dists_std", None), dict):
            stds = np.stack([np.asarray(r["dists_std"].get(m, np.full(G, np.nan))) for m in order], axis=0)

        # --- bars layout ---
        plt.figure(figsize=(16, 6))
        x = np.arange(M)
        group_width = 0.8
        bar_w = group_width / G
        start = -group_width/2 + bar_w/2

        # plot subgroup bars (each subgroup: dark -> light)
        for gi in range(G):
            xpos = x + start + gi * bar_w              # [M]
            y = means[:, gi]                           # [M]

            t = 0.0 if G <= 1 else (gi / (G - 1))      # 0..1 over subgroups
            amt = t * max_light                        # dark -> light
            colors = [lighten(base_rgb[mi], amt) for mi in range(M)]

            plt.bar(
                xpos, y,
                width=bar_w,
                color=colors,
                edgecolor="black", linewidth=0.3,
                capsize=3,
            )

        # --- mean line + deviation sticks + lengths ---
        mu = means.mean(axis=1)  # [M]

        # R-MAD (relative mean absolute deviation)
        eps = 1e-12
        mad = np.mean(np.abs(means - mu[:, None]), axis=1)     # [M]
        rgd = mad / (mu + eps)                                  # [M]

        for mi in range(M):
            left  = x[mi] - group_width/2
            right = x[mi] + group_width/2

            # mean line: black dashed (---)
            plt.hlines(mu[mi], left, right, color="black", linestyle="--", linewidth=2, zorder=5)

            # dotted vertical lines (...) to mean for each subgroup + label the length
            for gi in range(G):
                xpos = x[mi] + start + gi * bar_w
                yv = means[mi, gi]
                y0, y1 = (mu[mi], yv)

                # dotted line
                plt.plot([xpos, xpos], [y0, y1], linestyle=":", color="black", alpha=0.35, linewidth=1, zorder=6)

                # length label (absolute distance to mean), rotated top-to-down
                d = abs(yv - mu[mi])
                ymid = (y0 + y1) / 2.0
                plt.text(
                    xpos, ymid,
                    f"{d:.2f}",
                    rotation=90,
                    ha="center", va="center",
                    fontsize=dev_fontsize,
                    alpha=dev_alpha,
                    zorder=7
                )

            # annotate method summary above cluster
            top = np.nanmax(means[mi, :])
            plt.text(
                x[mi], top + 0.1,
                f"Avg: {mu[mi]:.2f}, R-MAD: {rgd[mi]:.2f}",
                ha="center", va="bottom",
                fontsize=15
            )

        # nicer method names (keep your order)
        pretty_methods = ['Full Dataset', 'DC', 'DC + FairDD', 'DC + TARAZ']
        if len(pretty_methods) == M:
            xt = pretty_methods
        else:
            xt = order

        plt.ylim(top=plt.ylim()[1] * 1.05)
        plt.xticks(x, xt, fontsize = 20)
        plt.yticks(fontsize = 20)
        # plt.xlabel("Method", fontsize = 20)
        # plt.ylabel(r"$||g_{all}-g_{group}||_2^2 \,/\, ||g_{all}||_2^2$")
        plt.grid(axis="y", alpha=0.5)
        # plt.ylabel(
        #     r"Normalized group training statistic deviation  ",
        #     fontsize=16
        # )

        plt.tight_layout()
        plt.savefig(f"{ds}_ipc{ipc_val}.pdf")

        # plt.close()
