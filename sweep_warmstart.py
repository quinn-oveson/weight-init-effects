import argparse
import csv
import hashlib
import os
import time

import torch
import torch.nn as nn

import sweep_config as C
from lorenz96 import diagnostics as G
from lorenz96 import metrics as M
from lorenz96.data import STRIDE
from lorenz96.models import CircularCNN, count_params
from lorenz96.stream import CycleStream
from lorenz96.system import DT

HOURS_PER_MTU = 120.0

CONFIG_COLS = ["arm", "warm_start", "data_all", "frozen", "lr", "noise", "hidden", "n_layers",
               "kernel", "history", "batch_size", "step_budget", "F", "n_params", "seed",
               "data_seed", "noise_seed", "init_seed", "shuffle_seed"]

TRAINING_COLS = CONFIG_COLS + [
    "cycle", "step", "cum_step", "cum_samples", "cycle_wall_s", "n_train", "train_loss",
    "val_mse_noisy", "val_mse_clean", "train_val_gap", "valid_time_mtu", "valid_time_lyap",
    "weight_norm"]

CURVE_COLS = CONFIG_COLS + [
    "cycle", "step", "cum_step", "lead_step", "lead_hours", "lead_mtu", "lead_lyapunov", "nrmse"]


def _template_model():
    return CircularCNN(hidden=C.HIDDEN, n_layers=C.N_LAYERS, kernel=C.KERNEL,
                       in_channels=C.HISTORY)


DIAG_COLS = G.diagnostic_columns(_template_model())

SUMMARY_COLS = CONFIG_COLS + [
    "cycle", "n_train", "cycle_steps", "cum_step", "cum_samples", "cycle_wall_s",
    "final_train_loss", "final_val_mse_noisy", "final_val_mse_clean", "final_train_val_gap",
    "best_val_mse_clean", "step_of_best", "rollout_std", "truth_std", "stable",
    "batch_order_hash"] + [f"vt_mtu@{t}" for t in C.THRESHOLDS] + DIAG_COLS


def slurm_ranges(ids):
    # Exact SLURM --array syntax; hand-writing ranges is how you submit the wrong tasks.
    ids, out, i = sorted(ids), [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(out)


def decode_task_id(task_id):
    if not 0 <= task_id < C.TOTAL_TASKS:
        raise SystemExit(f"task_id {task_id} out of range 0..{C.TOTAL_TASKS - 1}")
    return C.CELLS[task_id]


def resource_tier(cell):
    return (4, "01:00:00") if cell["frozen"] else (16, "10:00:00")


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Writer:
    def __init__(self, path, cols):
        self.cols = cols
        new = not os.path.exists(path)
        self.fh = open(path, "a", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=cols, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def write(self, row):
        # Round floats so binary-representation noise never reaches the CSV.
        self.w.writerow({k: round(v, 8) if isinstance(v, float) else v
                         for k, v in ((c, row.get(c, "")) for c in self.cols)})

    def close(self):
        self.fh.close()


@torch.no_grad()
def val_losses(model, val, dev):
    model.eval()
    pred = model(val.x.to(dev))
    return (float(nn.functional.mse_loss(pred, val.y.to(dev))),
            float(nn.functional.mse_loss(pred, val.y_clean.to(dev))))


def valid_time(curve, threshold):
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

    model, init_norms = None, None
    cum_step = cum_samples = 0
    lam = M.lambda1(C.F)
    eval_set = set(C.EVAL_STEPS)
    save_ckpt = C.CHECKPOINT and (C.CHECKPOINT_ARMS is None or cell["arm"] in C.CHECKPOINT_ARMS)

    for cycle in range(C.N_CYCLES):
        seeds = stream.seeds(cycle)
        data = stream.cycle_data(cycle, data_all=bool(cell["data_all"]))
        x, y = data.x.to(dev), data.y.to(dev)
        n_train = x.shape[0]

        if model is None or not cell["warm_start"]:
            torch.manual_seed(seeds["init_seed"])
            model = _template_model().to(dev)
            init_norms = G.layer_norms(model)

        base = dict(cell, hidden=C.HIDDEN, n_layers=C.N_LAYERS, kernel=C.KERNEL,
                    history=C.HISTORY, batch_size=C.BATCH_SIZE, step_budget=C.STEP_BUDGET,
                    F=C.F, n_params=count_params(model), **seeds)

        # frozen trains only at cycle 0; afterwards it is the do-nothing baseline.
        budget = 0 if (cell["frozen"] and cycle > 0) else C.STEP_BUDGET
        opt = torch.optim.Adam(model.parameters(), lr=cell["lr"])
        g = torch.Generator().manual_seed(seeds["shuffle_seed"])

        step, run_loss, n_batch, order_hash = 0, 0.0, 0, ""
        best_clean, best_step = float("inf"), -1
        t0 = time.time()

        def do_eval(step, train_loss, force_curve=False):
            nonlocal best_clean, best_step
            mse_n, mse_c = val_losses(model, stream.val, dev)
            if mse_c < best_clean:
                best_clean, best_step = mse_c, step
            r = M.evaluate(model, F=C.F, n_steps=C.ROLLOUT_STEPS, n_init=C.ROLLOUT_INITS_EVAL,
                           seed=stream.test_seed, history=C.HISTORY, init_noise=cell["noise"])
            wn = float(sum(v ** 2 for v in G.layer_norms(model).values()) ** 0.5)
            w_train.write(dict(base, cycle=cycle, step=step, cum_step=cum_step + step,
                               cum_samples=cum_samples + step * C.BATCH_SIZE,
                               cycle_wall_s=round(time.time() - t0, 3), n_train=n_train,
                               train_loss=train_loss, val_mse_noisy=mse_n, val_mse_clean=mse_c,
                               train_val_gap=mse_n - train_loss, valid_time_mtu=r["mtu"],
                               valid_time_lyap=r["lyapunov_times"], weight_norm=wn))
            if force_curve or step in curve_points:
                full = M.evaluate(model, F=C.F, n_steps=C.ROLLOUT_STEPS,
                                  n_init=C.ROLLOUT_INITS, seed=stream.test_seed,
                                  history=C.HISTORY, init_noise=cell["noise"])
                for s, v in enumerate(full["curve"].cpu().tolist()):
                    mtu = (s + 1) * STRIDE * DT
                    w_curve.write(dict(base, cycle=cycle, step=step, cum_step=cum_step + step,
                                       lead_step=s + 1, lead_hours=mtu * HOURS_PER_MTU,
                                       lead_mtu=mtu, lead_lyapunov=mtu * lam, nrmse=v))
                return full
            return None

        curve_points = set(C.EVAL_STEPS[::C.CURVE_EVERY])
        final_full = None

        if budget == 0:
            final_full = do_eval(0, float("nan"), force_curve=True)
        while step < budget:
            perm = torch.randperm(n_train, generator=g)
            if step == 0:
                order_hash = hashlib.sha1(perm.numpy().tobytes()).hexdigest()[:16]
            perm = perm.to(dev)
            model.train()
            # drop_last keeps every gradient step identical in size, so steps are exactly ~FLOPs.
            for i in range(0, n_train - C.BATCH_SIZE + 1, C.BATCH_SIZE):
                idx = perm[i:i + C.BATCH_SIZE]
                loss = nn.functional.mse_loss(model(x[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                run_loss += float(loss.detach())
                n_batch += 1
                step += 1
                if step in eval_set or step == budget:
                    out = do_eval(step, run_loss / max(n_batch, 1), force_curve=(step == budget))
                    final_full = out or final_full
                    run_loss, n_batch = 0.0, 0
                    model.train()
                if step >= budget:
                    break

        cycle_wall = time.time() - t0
        cum_step += step
        cum_samples += step * C.BATCH_SIZE

        if save_ckpt:
            os.makedirs(C.CKPT_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(C.CKPT_DIR, f"{tag}_cycle{cycle}.pt"))

        curve = final_full["curve"].cpu()
        diag = G.collect(model, stream.val.x[:512].to(dev), init_norms)
        mse_n, mse_c = val_losses(model, stream.val, dev)
        w_sum.write(dict(base, cycle=cycle, n_train=n_train, cycle_steps=step, cum_step=cum_step,
                         cum_samples=cum_samples, cycle_wall_s=round(cycle_wall, 3),
                         final_train_loss=run_loss / max(n_batch, 1) if n_batch else float("nan"),
                         final_val_mse_noisy=mse_n, final_val_mse_clean=mse_c,
                         final_train_val_gap=mse_n - (run_loss / n_batch if n_batch else 0.0),
                         best_val_mse_clean=best_clean, step_of_best=best_step,
                         rollout_std=final_full["pred_std"], truth_std=final_full["truth_std"],
                         stable=int(final_full["stable"]), batch_order_hash=order_hash,
                         **{f"vt_mtu@{t}": valid_time(curve, t) for t in C.THRESHOLDS}, **diag))

        print(f"  cycle {cycle}: n={n_train} steps={step} val_clean={mse_c:.5f} "
              f"best={best_clean:.5f}@{best_step} gap={mse_n - mse_c:+.5f} "
              f"rank={diag['effective_rank']:.1f} dormant={diag['dormant_frac']:.3f}", flush=True)

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
