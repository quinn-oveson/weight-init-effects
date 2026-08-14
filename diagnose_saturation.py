"""From-scratch n_train x hidden x noise sweep to tell task saturation apart from over-parameterisation."""
import argparse
import itertools
import os
import time

import torch
import torch.nn as nn

import sweep_config as C
from lorenz96 import metrics as M
from lorenz96.models import CircularCNN, count_params
from lorenz96.stream import CycleStream
from sweep_warmstart import Writer, device, split_losses

GRID = dict(n_train=[500, 1000, 2000, 5000, 10000, 20000],
            hidden=[64, 128, 512],
            noise=[0.0, 0.10],
            lr=[1e-3, 3e-3],
            seed=[0, 1, 2])

SUMMARY_COLS = [
    "n_train", "hidden", "noise", "lr", "seed", "n_params", "params_per_patch",
    "step_budget", "batch_size", "step_of_best", "best_at_boundary",
    "best_train_mse_noisy", "best_train_mse_clean", "best_val_mse_noisy",
    "best_val_mse_clean", "best_test_mse_noisy", "best_test_mse_clean",
    "gen_gap_clean", "train_vs_noise_floor", "noise_floor",
    "final_val_mse_clean", "valid_time_mtu", "valid_time_lyap", "stable", "wall_s"]

TRACE_COLS = ["n_train", "hidden", "noise", "lr", "seed", "step",
              "train_mse_clean", "val_mse_clean", "test_mse_clean"]


def run_one(n_train, hidden, noise, lr, seed, dev, w_trace):
    # One from-scratch fit. Selection on val, reporting on test, exactly as the sweep does.
    stream = CycleStream(seed=seed, noise=noise, n_cycles=1, chunk=n_train,
                         n_val=C.N_VAL, history=C.HISTORY, F=C.F)
    stream.assert_no_leakage()
    data = stream.cycle_data(0, data_all=True)
    x, y = data.x.to(dev), data.y.to(dev)
    n = x.shape[0]

    torch.manual_seed(101 + seed * 1000)
    model = CircularCNN(hidden=hidden, n_layers=C.N_LAYERS, kernel=C.KERNEL,
                        in_channels=C.HISTORY).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(500_003 + seed * 100_003)

    # Capped at N_VAL so every config is measured on the same number of samples.
    g_tr = torch.Generator().manual_seed(seed + 1)
    tr_idx = torch.randperm(n, generator=g_tr)[:min(C.N_VAL, n)]
    train_eval = type(data)(data.x[tr_idx], data.y[tr_idx], data.y_clean[tr_idx])

    eval_set, best = set(C.EVAL_STEPS), dict(val=float("inf"))
    best_state, step, t0 = None, 0, time.time()
    row = dict(n_train=n_train, hidden=hidden, noise=noise, lr=lr, seed=seed)

    def do_eval(step):
        nonlocal best, best_state
        v_n, v_c = split_losses(model, stream.val, dev)
        t_n, t_c = split_losses(model, train_eval, dev)
        e_n, e_c = split_losses(model, stream.test, dev)
        w_trace.write(dict(row, step=step, train_mse_clean=t_c, val_mse_clean=v_c,
                           test_mse_clean=e_c))
        if v_c < best["val"]:
            best = dict(val=v_c, step=step, val_noisy=v_n, train_noisy=t_n,
                        train_clean=t_c, test_noisy=e_n, test_clean=e_c)
            best_state = {k: v.detach().to("cpu", copy=True)
                          for k, v in model.state_dict().items()}
        return v_c

    do_eval(0)
    final_val = float("nan")
    while step < C.STEP_BUDGET:
        perm = torch.randperm(n, generator=g).to(dev)
        model.train()
        for i in range(0, n - C.BATCH_SIZE + 1, C.BATCH_SIZE):
            idx = perm[i:i + C.BATCH_SIZE]
            loss = nn.functional.mse_loss(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step in eval_set or step == C.STEP_BUDGET:
                final_val = do_eval(step)
                model.train()
            if step >= C.STEP_BUDGET:
                break
        if n < C.BATCH_SIZE:   # smaller than one batch: nothing would ever run
            break

    # Once, on the handoff model only: per-eval-step rollouts dominate the sweep's cost.
    if best_state is not None:
        model.load_state_dict(best_state)
    r = M.evaluate(model, F=C.F, n_steps=C.ROLLOUT_STEPS, n_init=C.ROLLOUT_INITS,
                   seed=stream.test_seed, history=C.HISTORY, init_noise=noise)

    floor = 2.0 * noise ** 2
    return dict(row, n_params=count_params(model),
                params_per_patch=round(count_params(model) / (n * 40), 4),
                step_budget=C.STEP_BUDGET, batch_size=C.BATCH_SIZE,
                step_of_best=best["step"],
                best_at_boundary=int(best["step"] > 0
                                     and C.STEP_BUDGET < 3 * best["step"]),
                best_train_mse_noisy=best["train_noisy"],
                best_train_mse_clean=best["train_clean"],
                best_val_mse_noisy=best["val_noisy"], best_val_mse_clean=best["val"],
                best_test_mse_noisy=best["test_noisy"],
                best_test_mse_clean=best["test_clean"],
                gen_gap_clean=best["val"] - best["train_clean"],
                # <1 means it fit below the noise floor, i.e. memorising rather than out of signal.
                train_vs_noise_floor=(best["train_noisy"] / floor if floor > 0
                                      else float("nan")),
                noise_floor=floor, final_val_mse_clean=final_val,
                valid_time_mtu=r["mtu"], valid_time_lyap=r["lyapunov_times"],
                stable=int(r["stable"]), wall_s=round(time.time() - t0, 2))


def groups():
    # Grouped, not one config per task: normalizer() integrates 100k RK4 steps per process.
    return [dict(noise=nz, seed=sd, hidden=h)
            for nz, sd, h in itertools.product(GRID["noise"], GRID["seed"],
                                               GRID["hidden"])]


GROUPS = groups()


def collect(out_dir):
    import csv
    import glob
    for kind, cols in (("summary", SUMMARY_COLS), ("trace", TRACE_COLS)):
        rows, files = [], sorted(glob.glob(os.path.join(out_dir, f"task*_{kind}.csv")))
        for f in files:
            with open(f, newline="") as fh:
                rows.extend(csv.DictReader(fh))
        if not rows:
            print(f"[{kind}] no task files in {out_dir}")
            continue
        with open(os.path.join(out_dir, f"saturation_{kind}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"[{kind}] {len(files)} task files -> {len(rows)} rows")
    # A task killed partway still leaves a valid short CSV, so count rows, not files.
    import re
    per_task, seen = len(GRID["n_train"]) * len(GRID["lr"]), {}
    for f in glob.glob(os.path.join(out_dir, "task*_summary.csv")):
        tid = int(re.search(r"task(\d+)_", os.path.basename(f)).group(1))
        with open(f, newline="") as fh:
            seen[tid] = sum(1 for _ in csv.DictReader(fh))
    bad = []
    for i, g in enumerate(GROUPS):
        if i not in seen:
            print(f"  MISSING task {i:03d} {g}")
            bad.append(i)
        elif seen[i] != per_task:
            print(f"  SHORT   task {i:03d} {g} ({seen[i]}/{per_task} configs)")
            bad.append(i)
    if bad:
        print(f"  resubmit with --array={','.join(str(i) for i in sorted(bad))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/diagnose")
    p.add_argument("--task_id", type=int, help="one (noise, seed, hidden) group")
    p.add_argument("--print_array_specs", action="store_true")
    p.add_argument("--collect", action="store_true",
                   help="concatenate finished task files into saturation_*.csv")
    for k, v in GRID.items():
        p.add_argument(f"--{k}", type=float if k in ("noise", "lr") else int,
                       nargs="+", default=v)
    args = p.parse_args()

    if args.print_array_specs:
        print(f"total_tasks={len(GROUPS)} results={args.out_dir}")
        print(f"  --array=0-{len(GROUPS) - 1}   "
              f"({len(GRID['n_train']) * len(GRID['lr'])} configs per task)")
        return
    if args.collect:
        return collect(args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    dev = device()
    if args.task_id is not None:
        if not 0 <= args.task_id < len(GROUPS):
            raise SystemExit(f"task_id out of range 0..{len(GROUPS) - 1}")
        grp = GROUPS[args.task_id]
        combos = [(n, grp["hidden"], grp["noise"], lr, grp["seed"])
                  for n, lr in itertools.product(args.n_train, args.lr)]
        tag = f"task{args.task_id:03d}"
        print(f"task {args.task_id}: {grp} on {dev.type}", flush=True)
    else:
        combos = list(itertools.product(args.n_train, args.hidden, args.noise,
                                        args.lr, args.seed))
        tag = "local"

    w_sum = Writer(os.path.join(args.out_dir, f"{tag}_summary.csv"), SUMMARY_COLS)
    w_trace = Writer(os.path.join(args.out_dir, f"{tag}_trace.csv"), TRACE_COLS)
    print(f"{len(combos)} configs; budget {C.STEP_BUDGET} steps, "
          f"{len(C.EVAL_STEPS)} eval points\n", flush=True)
    t0 = time.time()
    for i, (n, h, nz, lr, sd) in enumerate(combos):
        r = run_one(n, h, nz, lr, sd, dev, w_trace)
        print(f"[{i + 1}/{len(combos)}] n={n:<6} hidden={h:<4} noise={nz:<5} "
              f"lr={lr:<7g} seed={sd}  val={r['best_val_mse_clean']:.5f} "
              f"test={r['best_test_mse_clean']:.5f} train={r['best_train_mse_clean']:.5f} "
              f"@{r['step_of_best']:<5} vt={r['valid_time_lyap']:.2f} "
              f"({r['wall_s']:.0f}s)", flush=True)
        w_sum.write(r)
    for w in (w_sum, w_trace):
        w.close()
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {args.out_dir}/{tag}_*.csv")


if __name__ == "__main__":
    main()
