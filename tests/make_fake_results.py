import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sweep_config as C
import sweep_warmstart as S


def val_curve(step, floor, best_step, overfit):
    # U-shaped in log-step: falls to `floor` at best_step, then degrades.
    l = np.log(np.maximum(step, 1) / best_step)
    return floor * (1.0 + 3.0 * l ** 2 * (l < 0) + overfit * l ** 2 * (l >= 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/fake")
    p.add_argument("--warm-penalty", type=float, default=0.15,
                   help="per-cycle penalty for warm_all; negative plants warm-is-better")
    p.add_argument("--diverge", action="store_true",
                   help="plant inf/nan rollouts and stable=0 in late warm_all cycles")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    for task_id, cell in enumerate(C.CELLS):
        tag = f"task{task_id:03d}"
        wt = S.Writer(os.path.join(args.out, f"{tag}_training.csv"), S.TRAINING_COLS)
        wc = S.Writer(os.path.join(args.out, f"{tag}_curves.csv"), S.CURVE_COLS)
        ws = S.Writer(os.path.join(args.out, f"{tag}_summary.csv"), S.SUMMARY_COLS)
        base = dict(cell, hidden=C.HIDDEN, n_layers=C.N_LAYERS, kernel=C.KERNEL,
                    history=C.HISTORY, batch_size=C.BATCH_SIZE, step_budget=C.STEP_BUDGET,
                    F=C.F, n_params=2_633_729, data_seed=cell["seed"],
                    noise_seed=cell["seed"] + 9973, init_seed=101, shuffle_seed=500_003)
        cum = 0
        for cycle in range(C.N_CYCLES):
            n_train = C.CHUNK * (cycle + 1) if cell["data_all"] else C.CHUNK
            budget = 0 if (cell["frozen"] and cycle > 0) else C.STEP_BUDGET
            # warm_all accumulates a penalty with each successive warm start.
            pen = args.warm_penalty * cycle if cell["arm"] == "warm_all" else 0.0
            floor = cell["noise"] ** 2 * 0.4 * (1 + pen) * (1 - 0.03 * cycle)
            best_step = max(1, int(90 * (1 + 0.4 * cycle) * (0.5 if cell["warm_start"] else 1.0)))
            steps = [s for s in C.EVAL_STEPS if s <= max(budget, 1)] or [0]

            best_v, best_s = float("inf"), -1
            for s in steps:
                v = float(val_curve(s, floor, best_step, 6.0) * (1 + 0.02 * rng.standard_normal()))
                if v < best_v:
                    best_v, best_s = v, s
                wt.write(dict(base, cycle=cycle, step=s, cum_step=cum + s,
                              cum_samples=(cum + s) * C.BATCH_SIZE, cycle_wall_s=1.0,
                              n_train=n_train, train_loss=v * 0.5,
                              val_mse_noisy=v + 2 * cell["noise"] ** 2, val_mse_clean=v,
                              train_val_gap=v * 0.5 + 2 * cell["noise"] ** 2,
                              train_mse_noisy=v * 0.8 + 2 * cell["noise"] ** 2,
                              train_mse_clean=v * 0.8, test_mse_noisy=v * 1.05 + 2 * cell["noise"] ** 2,
                              test_mse_clean=v * 1.05,
                              valid_time_mtu=0.6 / (1 + 20 * v), valid_time_lyap=1.0 / (1 + 20 * v),
                              weight_norm=8.0 + 0.3 * cycle))

            # Only the end-of-budget model diverges, matching the real sweep.
            blown = (args.diverge and cell["arm"] == "warm_all" and cycle >= C.N_CYCLES // 2)

            def curve(step, basis, diverge):
                sk = float(val_curve(step, floor, best_step, 6.0))
                for lead in range(1, C.ROLLOUT_STEPS + 1):
                    mtu = lead * 0.05
                    nr = float(1 - np.exp(-(mtu * (1.7 + 40 * sk))))
                    if diverge and lead > C.ROLLOUT_STEPS // 3:
                        nr = float("inf") if lead % 2 else float("nan")
                    wc.write(dict(base, cycle=cycle, step=step, cum_step=cum + step,
                                  curve_basis=basis, lead_step=lead, lead_hours=mtu * 120,
                                  lead_mtu=mtu, lead_lyapunov=mtu * 1.671, nrmse=nr))

            for s in [x for x in steps if x in set(C.EVAL_STEPS[::C.CURVE_EVERY])] + [steps[-1]]:
                curve(s, "eval", blown and s == steps[-1])
            # Shares a step with an eval curve whenever best_s is on the curve grid.
            curve(best_s, "best", False)

            cum += budget
            diag = {c: 0.0 for c in S.DIAG_COLS}
            diag["effective_rank"] = 300 - (12 * cycle if cell["arm"] == "warm_all" else 0)
            diag["dormant_frac"] = 0.01 * cycle if cell["arm"] == "warm_all" else 0.0
            diag["weight_norm"] = 8.0 + 0.3 * cycle
            for n in [k for k in S.DIAG_COLS if k.startswith("wratio::")]:
                diag[n] = 1.0 + 0.05 * cycle
            # frozen never trains after cycle 0, so its train loss is legitimately empty.
            empty = cell["frozen"] and cycle > 0
            ws.write(dict(base, cycle=cycle, n_train=n_train, cycle_steps=budget, cum_step=cum,
                          cum_samples=cum * C.BATCH_SIZE, cycle_wall_s=280.0,
                          final_train_loss="" if empty else best_v * 2,
                          final_val_mse_noisy=best_v * 2.4,
                          final_val_mse_clean=best_v * 2.4,
                          final_train_val_gap="" if empty else best_v * 0.4,
                          final_train_mse_clean=best_v * 1.9, final_test_mse_noisy=best_v * 2.5,
                          final_test_mse_clean=best_v * 2.45,
                          final_gen_gap_clean=best_v * 2.4 - best_v * 1.9,
                          best_train_mse_clean=best_v * 0.8, best_test_mse_noisy=best_v * 1.1,
                          best_test_mse_clean=best_v * 1.05,
                          best_val_mse_clean=best_v, step_of_best=best_s,
                          best_at_boundary=int(budget < 3 * max(best_s, 1)),
                          rollout_std=float("inf") if blown else 0.99, truth_std=1.02,
                          stable=int(not blown), best_stable=1, best_rollout_std=0.99,
                          batch_order_hash=f"{cell['seed']:016x}", **diag,
                          **{f"vt_mtu@{t}": 0.3 + 0.2 * t for t in C.THRESHOLDS},
                          **{f"vt_best_mtu@{t}": 0.35 + 0.2 * t for t in C.THRESHOLDS}))
        for w in (wt, wc, ws):
            w.close()
    print(f"wrote {len(C.CELLS)} fake tasks to {args.out} "
          f"(warm_all penalty {args.warm_penalty}/cycle"
          + (", divergence planted)" if args.diverge else ")"))


if __name__ == "__main__":
    main()
