import os

# The preset decides the grid AND the output directory, so two presets never collide.
PRESET = os.environ.get("L96_SWEEP", "warmstart")

SEEDS = [0, 1, 2, 3, 4]          # Hard-coded for replicability, not merely recorded.
COLD_LR = 1e-3
WARM_LRS = [1e-3, 3e-4, 1e-4, 3e-5]   # Largest warm LR equals COLD_LR by construction.
NOISE_LEVELS = [0.05, 0.10]      # ~hard-but-solvable, and a poorly-observed-region level.

ARMS = [
    # (name, warm_start, data_all, learning rates)
    ("cold", 0, 1, [COLD_LR]),
    ("warm_all", 1, 1, WARM_LRS),
    ("warm_new", 1, 0, WARM_LRS),
]

N_CYCLES = 10
CHUNK = 2_000
N_VAL = 4_000
EPOCHS = 120                     # Generous: train well past convergence, stop post hoc.
BATCH_SIZE = 256
HIDDEN = 512                     # ~2.63M params vs 800k training patches (overparameterized).
N_LAYERS = 3
KERNEL = 5
HISTORY = 4
F = 8.0
CURVE_EVERY = 20                 # Full forecast curves this often, plus every cycle end.
ROLLOUT_STEPS = 120
ROLLOUT_INITS = 64               # Full curves at cycle end.
ROLLOUT_INITS_EPOCH = 16         # Cheaper per-epoch scalar, keeps eval from dominating runtime.
THRESHOLDS = [0.2, 0.3, 0.5, 0.8]

PRESETS = {
    "warmstart": {},
    "smoke": dict(SEEDS=[0], NOISE_LEVELS=[0.05], WARM_LRS=[1e-3, 1e-4], N_CYCLES=3,
                  CHUNK=500, N_VAL=800, EPOCHS=5, HIDDEN=64, CURVE_EVERY=2,
                  ROLLOUT_STEPS=40, ROLLOUT_INITS=8),
}

for _k, _v in PRESETS.get(PRESET, {}).items():
    globals()[_k] = _v
if PRESET == "smoke":
    ARMS = [("cold", 0, 1, [COLD_LR]), ("warm_all", 1, 1, WARM_LRS),
            ("warm_new", 1, 0, WARM_LRS)]

RESULTS_DIR = os.path.join("results", PRESET)


def cells():
    # One cell = one (arm, lr, noise, seed) = one full sequential cycle stream = one SLURM task.
    out = []
    for noise in NOISE_LEVELS:
        for arm, warm, data_all, lrs in ARMS:
            for lr in lrs:
                for seed in SEEDS:
                    out.append(dict(arm=arm, warm_start=warm, data_all=data_all, lr=lr,
                                    noise=noise, seed=seed))
    return out


CELLS = cells()
TOTAL_TASKS = len(CELLS)
