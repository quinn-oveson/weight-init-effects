import argparse
import csv
import glob
import os
import sys
from collections import Counter, defaultdict

import sweep_config as C

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

    # cold and warm_all train on the same data, so their per-cycle step counts must match.
    steps = defaultdict(dict)
    for r in summary_rows:
        if r["arm"] in ("cold", "warm_all"):
            steps[(float(r["noise"]), int(r["seed"]), int(r["cycle"]))][r["arm"]] = \
                int(r["cycle_steps"])
    for k, d in steps.items():
        if len(d) == 2 and d["cold"] != d["warm_all"]:
            problems.append(f"step mismatch at noise/seed/cycle {k}: {d}")

    unstable = [(r["arm"], r["seed"], r["cycle"]) for r in summary_rows if r["stable"] != "1"]
    if unstable:
        problems.append(f"{len(unstable)} unstable rollouts, e.g. {unstable[:3]}")

    gaps = [float(r["final_train_val_gap"]) for r in summary_rows]
    if gaps:
        problems.append(f"INFO train_val_gap range [{min(gaps):+.5f}, {max(gaps):+.5f}] "
                        f"-- if never positive, the model is not overfitting and the "
                        f"phenomenon cannot appear")
    return problems


def check_curves(curve_rows, sample=200):
    # NRMSE should increase with lead time; flag configurations where it does not.
    by_run = defaultdict(list)
    for r in curve_rows:
        key = (r["arm"], r["lr"], r["noise"], r["seed"], r["cycle"], r["epoch"])
        by_run[key].append((int(r["lead_step"]), float(r["nrmse"])))
    bad = []
    for key, pts in list(by_run.items())[:sample]:
        pts.sort()
        v = [p[1] for p in pts]
        if v[-1] < v[0]:
            bad.append(key)
    return bad


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
                                    int(r.get("epoch", 0) or 0),
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

    print(f"\ntables -> {out_dir}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
