

#!/usr/bin/env python3
import os
import copy
import argparse
import random
import numpy as np
import torch

from utils import (
    get_loops,
    get_dataset,
    get_network,
    get_eval_pool,
    get_daparam,
    ParamDiffAug,
)

from utils_draw import (
    load_synthetic_checkpoint,
    extract_syn_data,
    train_from_init_state,
    per_group_dispersion_distances,
    align_to_reference,
    plot_groups_x_methods_bars,
    plot_methods_x_groups_bars,
)

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
    

    parser.add_argument("--num_full", type=int, default=4) # how many full-training runs to eval
    parser.add_argument("--num_eval", type=int, default=20) # how many synthetic-training runs to eval

    args = parser.parse_args()

     

    
    for dataset in [
                    "CIFAR10_S_90",
                    "Colored_FashionMNIST_foreground",
                    "Colored_FashionMNIST_background",
                    "Colored_MNIST_foreground",
                    # "Colored_MNIST_background",
                    "UTKface"
                    ]:
        args.dataset = dataset
        for ipc in [100]:
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
            

            order = ("Full", "NoFair", "FairDD", "NoOrtho")
            per_eval_dists = {k: [] for k in order}
            norms = {k: [] for k in order}
            gids_ref = None

            # --------------------------
            # 1) FULL: only num_full runs
            # --------------------------
            for it_full in range(args.num_full):
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

                print(f"Completed Full eval run {it_full+1}/4")

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
                    # print(f"Loaded {fair_crt} from: {path_used}")

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



                if it_eval % 5 == 0:
                    print(f"Completed eval run {it_eval+1}/{args.num_eval}")

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
