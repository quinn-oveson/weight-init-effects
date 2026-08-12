import argparse
import csv
import os
import time

import torch
import torch.nn as nn

import sweep_config as C
from lorenz96 import metrics as M
from lorenz96.data import STRIDE
from lorenz96.models import CircularCNN, count_params
from lorenz96.stream import CycleStream
from lorenz96.system import DT

HOURS_PER_MTU = 120.0

CONFIG_COLS = ["arm", "warm_start", "data_all", "lr", "noise", "hidden", "n_layers", "kernel",
               "history", "batch_size", "F", "n_params", "seed", "data_seed", "noise_seed",
               "init_seed", "shuffle_seed"]

TRAINING_COLS = CONFIG_COLS + [
    "cycle", "epoch", "cum_epochs", "cycle_steps", "cum_steps", "cycle_samples", "cum_samples",
    "cycle_wall_s", "cum_wall_s", "n_train", "train_loss", "val_mse_noisy", "val_mse_clean",
    "train_val_gap", "valid_time_mtu", "valid_time_lyap"]

CURVE_COLS = CONFIG_COLS + [
    "cycle", "epoch", "cum_steps", "lead_step", "lead_hours", "lead_mtu", "lead_lyapunov",
    "nrmse"]

SUMMARY_COLS = CONFIG_COLS + [
    "cycle", "n_train", "final_train_loss", "final_val_mse_noisy", "final_val_mse_clean",
    "final_train_val_gap", "best_val_mse_clean", "epoch_of_best", "rollout_std", "truth_std",
    "stable", "cycle_steps", "cum_steps", "cycle_samples", "cum_samples", "cycle_wall_s",
    "cum_wall_s", "weight_norm"] + [f"vt_mtu@{t}" for t in C.THRESHOLDS]


def decode_task_id(task_id):
    if not 0 <= task_id < C.TOTAL_TASKS:
        raise SystemExit(f"task_id {task_id} out of range 0..{C.TOTAL_TASKS - 1}")
    return C.CELLS[task_id]


def slurm_ranges(ids):
    # Exact SLURM --array syntax; hand-writing ranges is the documented way to submit wrong tasks.
    ids, out, i = sorted(ids), [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(out)


def resource_tier(cell):
    # warm_new trains on one chunk per cycle; the accumulating arms grow linearly and cost more.
    return (8, "02:00:00") if cell["data_all"] == 0 else (16, "08:00:00")


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Writer:
    # Append-only CSV with a header written once.
    def __init__(self, path, cols):
        self.cols = cols
        new = not os.path.exists(path)
        self.fh = open(path, "a", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=cols, extrasaction="raise")
        if new:
            self.w.writeheader()

    def write(self, row):
        # Round floats so binary-representation noise never reaches the CSV.
        self.w.writerow({k: round(v, 8) if isinstance(v, float) else row.get(k, "")
                         for k in self.cols for v in [row.get(k, "")]})

    def close(self):
        self.fh.close()


@torch.no_grad()
def val_losses(model, val, dev):
    model.eval()
    pred = model(val.x.to(dev))
    return (float(nn.functional.mse_loss(pred, val.y.to(dev))),
            float(nn.functional.mse_loss(pred, val.y_clean.to(dev))))


def rollout_metrics(model, stream, noise, n_init, n_steps):
    res = M.evaluate(model, F=stream.F, n_steps=n_steps, n_init=n_init, seed=stream.test_seed,
                     history=stream.history, init_noise=noise)
    return res


def valid_time_from_curve(curve, threshold):
    over = (curve > threshold).nonzero()
    steps = int(over[0]) if len(over) else len(curve)
    return steps * STRIDE * DT


def run_cell(cell, task_id, outdir):
    dev = device()
    torch.manual_seed(cell["seed"])
    stream = CycleStream(seed=cell["seed"], noise=cell["noise"], n_cycles=C.N_CYCLES,
                         chunk=C.CHUNK, n_val=C.N_VAL, history=C.HISTORY, F=C.F)
    stream.assert_no_leakage()

    tag = f"task{task_id:03d}"
    w_train = Writer(os.path.join(outdir, f"{tag}_training.csv"), TRAINING_COLS)
    w_curve = Writer(os.path.join(outdir, f"{tag}_curves.csv"), CURVE_COLS)
    w_sum = Writer(os.path.join(outdir, f"{tag}_summary.csv"), SUMMARY_COLS)

    model = None
    cum_steps = cum_samples = cum_epochs = 0
    cum_wall = 0.0
    lam = M.lambda1(C.F)

    for cycle in range(C.N_CYCLES):
        seeds = stream.seeds(cycle)
        data = stream.cycle_data(cycle, data_all=bool(cell["data_all"]))
        x, y = data.x.to(dev), data.y.to(dev)
        n_train = x.shape[0]

        if model is None or not cell["warm_start"]:
            torch.manual_seed(seeds["init_seed"])
            model = CircularCNN(hidden=C.HIDDEN, n_layers=C.N_LAYERS, kernel=C.KERNEL,
                                in_channels=C.HISTORY).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=cell["lr"])

        base = dict(cell, hidden=C.HIDDEN, n_layers=C.N_LAYERS, kernel=C.KERNEL,
                    history=C.HISTORY, batch_size=C.BATCH_SIZE, F=C.F,
                    n_params=count_params(model), **seeds)

        g = torch.Generator().manual_seed(seeds["shuffle_seed"])
        cycle_steps = cycle_samples = 0
        cycle_wall = 0.0
        best_clean, best_epoch = float("inf"), -1

        for epoch in range(C.EPOCHS):
            t0 = time.time()
            model.train()
            perm = torch.randperm(n_train, generator=g).to(dev)
            run_loss, n_batch = 0.0, 0
            # drop_last keeps every gradient step the same size, so steps are exactly ~FLOPs.
            for i in range(0, n_train - C.BATCH_SIZE + 1, C.BATCH_SIZE):
                idx = perm[i:i + C.BATCH_SIZE]
                loss = nn.functional.mse_loss(model(x[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                run_loss += float(loss.detach())
                n_batch += 1
            cycle_wall += time.time() - t0
            cycle_steps += n_batch
            cycle_samples += n_batch * C.BATCH_SIZE
            train_loss = run_loss / max(n_batch, 1)

            mse_noisy, mse_clean = val_losses(model, stream.val, dev)
            r = rollout_metrics(model, stream, cell["noise"], C.ROLLOUT_INITS_EPOCH,
                                C.ROLLOUT_STEPS)
            if mse_clean < best_clean:
                best_clean, best_epoch = mse_clean, epoch

            w_train.write(dict(base, cycle=cycle, epoch=epoch, cum_epochs=cum_epochs + epoch + 1,
                               cycle_steps=cycle_steps, cum_steps=cum_steps + cycle_steps,
                               cycle_samples=cycle_samples, cum_samples=cum_samples + cycle_samples,
                               cycle_wall_s=round(cycle_wall, 3),
                               cum_wall_s=round(cum_wall + cycle_wall, 3), n_train=n_train,
                               train_loss=train_loss, val_mse_noisy=mse_noisy,
                               val_mse_clean=mse_clean, train_val_gap=mse_noisy - train_loss,
                               valid_time_mtu=r["mtu"], valid_time_lyap=r["lyapunov_times"]))

            last = epoch == C.EPOCHS - 1
            if last or (epoch + 1) % C.CURVE_EVERY == 0:
                full = rollout_metrics(model, stream, cell["noise"], C.ROLLOUT_INITS,
                                       C.ROLLOUT_STEPS)
                curve = full["curve"].cpu()
                for s, v in enumerate(curve.tolist()):
                    mtu = (s + 1) * STRIDE * DT
                    w_curve.write(dict(base, cycle=cycle, epoch=epoch,
                                       cum_steps=cum_steps + cycle_steps, lead_step=s + 1,
                                       lead_hours=mtu * HOURS_PER_MTU, lead_mtu=mtu,
                                       lead_lyapunov=mtu * lam, nrmse=v))
                if last:
                    final_full = full

        cum_steps += cycle_steps
        cum_samples += cycle_samples
        cum_epochs += C.EPOCHS
        cum_wall += cycle_wall

        curve = final_full["curve"].cpu()
        vts = {f"vt_mtu@{t}": valid_time_from_curve(curve, t) for t in C.THRESHOLDS}
        wnorm = float(sum(p.detach().pow(2).sum() for p in model.parameters()).sqrt())
        w_sum.write(dict(base, cycle=cycle, n_train=n_train, final_train_loss=train_loss,
                         final_val_mse_noisy=mse_noisy, final_val_mse_clean=mse_clean,
                         final_train_val_gap=mse_noisy - train_loss, best_val_mse_clean=best_clean,
                         epoch_of_best=best_epoch, rollout_std=final_full["pred_std"],
                         truth_std=final_full["truth_std"], stable=int(final_full["stable"]),
                         cycle_steps=cycle_steps, cum_steps=cum_steps,
                         cycle_samples=cycle_samples, cum_samples=cum_samples,
                         cycle_wall_s=round(cycle_wall, 3), cum_wall_s=round(cum_wall, 3),
                         weight_norm=wnorm, **vts))

        print(f"  cycle {cycle}: n={n_train} steps={cycle_steps} train={train_loss:.5f} "
              f"val_clean={mse_clean:.5f} gap={mse_noisy - train_loss:+.5f} "
              f"vt={final_full['lyapunov_times']:.2f}L", flush=True)

    for w in (w_train, w_curve, w_sum):
        w.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", type=int)
    p.add_argument("--print_array_specs", action="store_true")
    args = p.parse_args()

    if args.print_array_specs:
        tiers = {}
        for i, cell in enumerate(C.CELLS):
            tiers.setdefault(resource_tier(cell), []).append(i)
        print(f"preset={C.PRESET} total_tasks={C.TOTAL_TASKS} results={C.RESULTS_DIR}")
        for (mem, t), ids in sorted(tiers.items(), key=lambda kv: -kv[0][0]):
            print(f"  mem={mem}G time={t} n_tasks={len(ids)}  --array={slurm_ranges(ids)}")
        return

    if args.task_id is None:
        p.error("--task_id is required (unless --print_array_specs is given)")

    cell = decode_task_id(args.task_id)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    print(f"task {args.task_id}: {cell} on {device().type}", flush=True)
    t0 = time.time()
    run_cell(cell, args.task_id, C.RESULTS_DIR)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
