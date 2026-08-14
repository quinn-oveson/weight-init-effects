#!/bin/bash --login
# SLURM array job for the cold-vs-frozen saturation diagnostic (diagnose_saturation.py).
#
# WHY THIS EXISTS: in the first E1/E2 sweep the cold arm never beat frozen -- retraining
# from scratch on 2k->20k samples gave the same val MSE as the one-shot 2k model. Either
# the task saturates at one chunk, or HIDDEN=512 is over-parameterised so early stopping
# caps skill regardless of data. This sweeps n_train x hidden x noise from scratch to
# tell those apart. The CHUNK/HIDDEN choice for the E2 re-run is blocked on the answer.
#
# One array task = one (noise, seed, hidden) group = 12 configs (n_train x lr) run in
# sequence. Grouped rather than one-config-per-task because normalizer() integrates
# 100k RK4 steps in every process and val/test trajectories cache per (seed, noise) --
# one task per config would pay that fixed cost 216 times instead of 18.
#
# Preset is required, same contract as warmstart_array.sbatch: it sets STEP_BUDGET and
# EVAL_STEPS, so the diagnostic uses the identical stopping grid as the sweep and its
# answer transfers directly.
#   sbatch --export=ALL,L96_SWEEP=e1e2 --constraint='hopper|lovelace' \
#          --array=0-17 slurm/diagnose_array.sbatch
#
# DON'T hand-write --array. Ask for it:
#   L96_SWEEP=e1e2 python diagnose_saturation.py --print_array_specs
#
# GPU SELECTION: see the long rationale in slurm/warmstart_array.sbatch. Short version --
# always pass --constraint='hopper|lovelace'. P100/m9g nodes are Broadwell-era and a
# meaningful part of this job is CPU-bound NumPy RK4 integration that a faster GPU cannot
# hide. This job is MORE CPU-bound than the sweep: it builds a fresh trajectory for every
# n_train value and runs only one rollout per config instead of one per eval step.
#
# NOTE the time budget is NOT the sweep's 00:30:00: one task here runs 12 configs back to
# back, so its cost profile is unrelated to a single sweep cell. Tighten from
#   sacct -j <jobid> --format=JobID,MaxRSS,Elapsed,State
# on the run that already completed.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1     # Nothing here is CPU-parallel; 1 core schedules sooner.
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00       # 12 configs per task; see the NOTE above.
#SBATCH --array=0-17          # PLACEHOLDER -- confirm with --print_array_specs.
#SBATCH -J l96_diagnose
#SBATCH --output=slurm_logs/diagnose_%A_%a.out
#SBATCH --error=slurm_logs/diagnose_%A_%a.err

# mkdir -p slurm_logs before submitting -- SLURM will not create it for you.

# Fail loudly rather than silently running the default preset into the wrong directory.
: "${L96_SWEEP:?not set -- submit with --export=ALL,L96_SWEEP=e1e2}"

mamba activate weight-init-effects

cd "$SLURM_SUBMIT_DIR"   # the repo root you ran sbatch from (NOT $(dirname "$0") --
                          # SLURM copies this script to /var/spool/slurmd before running it)

# Decode the group through the same source of truth the run uses, rather than
# duplicating the grid here. Address the array ELEMENT, not $SLURM_JOB_ID, which can
# land the rename on a sibling task.
CELL_NAME=$(python -c "
from diagnose_saturation import GROUPS
g = GROUPS[${SLURM_ARRAY_TASK_ID}]
print(f\"diag_h{g['hidden']}_noise{g['noise']:g}_seed{g['seed']}\")
")
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="$CELL_NAME" || true

python diagnose_saturation.py --task_id "$SLURM_ARRAY_TASK_ID"

# ============================================================================
# ROLLOUT
# 1. Submit the whole array in one go:
#      mkdir -p slurm_logs
#      sbatch --export=ALL,L96_SWEEP=e1e2 --constraint='hopper|lovelace' \
#             --array=0-17 slurm/diagnose_array.sbatch
#    Watch:  squeue -u $USER
#
# 2. Concatenate when everything finishes (warns about any missing or short task):
#      python diagnose_saturation.py --collect
#
# RE-RUNNING FAILURES
# Tasks are independent and each writes only results/diagnose/task<NNN>_*.csv, and Writer
# truncates on open, so a failed task is resubmitted with --array=<id> and needs no
# cleanup first.
#
# READING THE RESULT (decision rule, fixed in advance)
#   best_val_mse_clean falls with n_train at hidden 64/128 but is flat at 512
#       -> over-parameterisation; narrow the model for the E2 re-run.
#   Flat at every width, and train_vs_noise_floor ~= 1 at the optimum
#       -> genuine saturation; shrink CHUNK so early cycles are data-limited.
#   Flat at noise 0.10 but falling at noise 0.0
#       -> the observation-noise floor binds; revisit noise levels, not architecture.
# train_vs_noise_floor < 1 means the model fit BELOW the observation noise, i.e. it is
# memorising noise rather than running out of signal.
# ============================================================================
