import argparse
import csv
import glob
import os
import sys
from collections import Counter, defaultdict

import sweep_config as C
from lorenz96.data import STRIDE
from lorenz96.metrics import lambda1
from lorenz96.system import DT

LAM = lambda1(C.F)

KINDS = ("training", "curves", "summary")


def model_key(row):
    # A "model type" is one configuration; seeds are the rows within its table.
    return (row["arm"], float(row["lr"]), float(row["noise"]))


def model_name(key):
    arm, lr, noise = key
    return f"{arm}_lr{lr:g}_noise{noise:g}"


def read_tasks(results_dir, kind):
    rows, files = [], sorted(glob.glob(os.path.join(results_dir, f"task*_{kind}.csv")))
    for f in files:
        with open(f, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows, files


def write_table(path, rows, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def check_completeness(summary_rows):
    # Every (arm, lr, noise, seed) cell should contribute exactly N_CYCLES summary rows.
    seen = Counter((r["arm"], float(r["lr"]), float(r["noise"]), int(r["seed"]))
                   for r in summary_rows)
    expected = {(c["arm"], c["lr"], c["noise"], c["seed"]) for c in C.CELLS}
    missing = sorted(expected - set(seen))
    partial = sorted(k for k, n in seen.items() if n != C.N_CYCLES)
    extra = sorted(set(seen) - expected)
    return missing, partial, extra


def check_invariants(summary_rows):
    problems = []

    # cold and warm_all must differ ONLY in initialization: same steps, same batch order.
    cells = defaultdict(dict)
    for r in summary_rows:
        if r["arm"] in ("cold", "warm_all"):
            key = (float(r["noise"]), float(r["lr"]), int(r["seed"]), int(r["cycle"]))
            cells[key][r["arm"]] = (int(r["cycle_steps"]), r["batch_order_hash"])
    for k, d in cells.items():
        if len(d) == 2:
            if d["cold"][0] != d["warm_all"][0]:
                problems.append(f"step-count mismatch at {k}: {d['cold'][0]} vs {d['warm_all'][0]}")
            if d["cold"][1] != d["warm_all"][1]:
                problems.append(f"batch-order mismatch at {k} — more than init differs")

    # The ramp only measures warm-starting if it finishes before the optimum.
    ramp = [r for r in summary_rows if r["arm"].endswith("_lr_ramp") and r.get("step_of_best")]
    if ramp:
        inside = sum(1 for r in ramp if int(r["step_of_best"]) < C.WARMUP_STEPS)
        problems.append(f"INFO {inside}/{len(ramp)} lr-ramp cycles peaked inside the "
                        f"{C.WARMUP_STEPS}-step warmup -- shorten WARMUP_STEPS if most did")

    unstable = [(r["arm"], r["seed"], r["cycle"]) for r in summary_rows if r["stable"] != "1"]
    if unstable:
        problems.append(f"{len(unstable)} unstable rollouts, e.g. {unstable[:3]}")

    # Must be an eval()-mode train MSE: final_train_val_gap degenerates to val error alone.
    col = ("final_gen_gap_clean" if any(r.get("final_gen_gap_clean") for r in summary_rows)
           else None)
    if col is None:
        problems.append("INFO no final_gen_gap_clean column -- overfitting cannot be "
                        "checked on this sweep; re-run to log train/test MSE")
    else:
        gaps = [float(r[col]) for r in summary_rows if r.get(col) not in (None, "")]
        if gaps:
            problems.append(f"INFO gen_gap_clean (val - train) range "
                            f"[{min(gaps):+.5f}, {max(gaps):+.5f}] -- if never positive, "
                            f"the model is not overfitting and the phenomenon cannot appear")
    return problems


def check_curves(curve_rows, sample=200):
    # NRMSE should increase with lead time; flag configurations where it does not.
    by_run = defaultdict(list)
    for r in curve_rows:
        # In the key because a "best" curve can share a step with an "eval" curve.
        key = (r["arm"], r["lr"], r["noise"], r["seed"], r["cycle"], r["step"],
               r.get("curve_basis", "eval"))
        by_run[key].append((int(r["lead_step"]), float(r["nrmse"])))
    bad = []
    for key, pts in list(by_run.items())[:sample]:
        pts.sort()
        v = [p[1] for p in pts]
        if v[-1] < v[0]:
            bad.append(key)
    return bad


def gap_vs_lead(curve_rows, summary_rows, out_path):
    # Per-seed warm_all-minus-cold NRMSE vs lead on both bases; never compare on "final" alone.
    best_step = {}
    for r in summary_rows:
        key = (r["arm"], float(r["noise"]), float(r["lr"]), int(r["seed"]), int(r["cycle"]))
        best_step[key] = int(r["step_of_best"])

    # curve_basis="best" carries an exact handoff curve; older sweeps must snap to nearest.
    curves = defaultdict(lambda: defaultdict(dict))
    exact = defaultdict(dict)
    for r in curve_rows:
        if r["arm"] not in ("cold", "warm_all"):
            continue
        key = (r["arm"], float(r["noise"]), float(r["lr"]), int(r["seed"]), int(r["cycle"]))
        if r.get("curve_basis") == "best":
            exact[key][int(r["lead_step"])] = float(r["nrmse"])
        else:
            curves[key][int(r["step"])][int(r["lead_step"])] = float(r["nrmse"])

    def pick(key, basis):
        if basis == "best" and key in exact:
            return best_step.get(key), exact[key]
        steps = sorted(curves.get(key, {}))
        if not steps:
            return None, None
        if basis == "final":
            chosen = steps[-1]
        else:
            target = best_step.get(key)
            if target is None:
                return None, None
            chosen = min(steps, key=lambda st: abs(st - target))
        return chosen, curves[key][chosen]

    cells = {k[1:] for k in curves if k[0] == "cold"} & {k[1:] for k in curves if k[0] == "warm_all"}
    rows = []
    for basis in ("best", "final"):
        for noise, lr, seed, cycle in sorted(cells):
            s_c, c_cold = pick(("cold", noise, lr, seed, cycle), basis)
            s_w, c_warm = pick(("warm_all", noise, lr, seed, cycle), basis)
            if c_cold is None or c_warm is None:
                continue
            for lead in sorted(set(c_cold) & set(c_warm)):
                mtu = lead * STRIDE * DT
                rows.append(dict(basis=basis, noise=noise, lr=lr, seed=seed, cycle=cycle,
                                 lead_step=lead, lead_mtu=round(mtu, 8),
                                 lead_hours=round(mtu * 120.0, 8),
                                 lead_lyapunov=round(mtu * LAM, 8),
                                 cold_step=s_c, warm_all_step=s_w,
                                 cold_nrmse=round(c_cold[lead], 8),
                                 warm_all_nrmse=round(c_warm[lead], 8),
                                 gap=round(c_warm[lead] - c_cold[lead], 8)))
    if rows:
        write_table(out_path, rows, list(rows[0]))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=C.RESULTS_DIR)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "by_model")

    ok = True
    for kind in KINDS:
        rows, files = read_tasks(args.results_dir, kind)
        if not rows:
            print(f"[{kind}] no task files found in {args.results_dir}")
            ok = False
            continue
        cols = list(rows[0].keys())

        groups = defaultdict(list)
        for r in rows:
            groups[model_key(r)].append(r)

        for key, grp in sorted(groups.items()):
            grp.sort(key=lambda r: (int(r["seed"]), int(r["cycle"]),
                                    int(r.get("step", 0) or 0),
                                    int(r.get("lead_step", 0) or 0)))
            write_table(os.path.join(out_dir, f"{model_name(key)}_{kind}.csv"), grp, cols)

        write_table(os.path.join(args.results_dir, f"all_{kind}.csv"), rows, cols)
        print(f"[{kind}] {len(files)} task files -> {len(rows)} rows, "
              f"{len(groups)} model types")

        if kind == "summary":
            missing, partial, extra = check_completeness(rows)
            if missing:
                print(f"  MISSING {len(missing)} cells, e.g. {missing[:3]}")
                ok = False
            if partial:
                print(f"  PARTIAL {len(partial)} cells (wrong cycle count), e.g. {partial[:3]}")
                ok = False
            if extra:
                print(f"  UNEXPECTED cells: {extra[:3]}")
                ok = False
            if not (missing or partial or extra):
                print(f"  complete: {len(C.CELLS)} cells x {C.N_CYCLES} cycles")
            for msg in check_invariants(rows):
                print(f"  {msg}")
                if not msg.startswith("INFO"):
                    ok = False

        if kind == "curves":
            bad = check_curves(rows)
            if bad:
                print(f"  WARN {len(bad)} runs where NRMSE does not increase with lead time, "
                      f"e.g. {bad[:2]}")
            srows, _ = read_tasks(args.results_dir, "summary")
            gap_rows = gap_vs_lead(rows, srows,
                                   os.path.join(args.results_dir, "gap_vs_lead.csv"))
            n_best = sum(1 for r in gap_rows if r["basis"] == "best")
            print(f"  gap-vs-lead (headline): {len(gap_rows)} rows "
                  f"({n_best} best-stopped, {len(gap_rows)-n_best} final) -> gap_vs_lead.csv")

    print(f"\ntables -> {out_dir}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
