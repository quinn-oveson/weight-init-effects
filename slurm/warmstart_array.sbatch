#!/bin/bash --login
# SLURM array job for the warm-start degradation sweep (sweep_warmstart.py).
#
# WHICH SWEEP THIS RUNS IS NOT IN THIS FILE. Name a sweep_config.py preset via
# the L96_SWEEP environment variable on every submission:
#   sbatch --export=ALL,L96_SWEEP=e1e2 ... slurm/warmstart_array.sbatch
# The preset decides the grid AND the output directory (results/<preset>/), so
# two presets can sit in the queue without colliding. It is an env var rather
# than an edit to sweep_config.py because each compute node re-imports that
# module when its task actually STARTS -- editing it while an array is queued
# would silently change what that queued job runs.
#
# One array task = one (arm, lr, noise, seed) cell = the whole 10-cycle chain.
# Cycles are inherently sequential (warm start depends on the previous cycle),
# so they cannot be split across tasks.
#
# DON'T hand-write --array ranges. Ask for them:
#   L96_SWEEP=e1e2 python sweep_warmstart.py --print_array_specs
# One tier covers every cell: 8G and 00:30:00, from the measured 3.0 min/cell at
# 10 cycles and MaxRSS 1.3-2.1GB on the first run.
#
# Budget is in optimizer STEPS, not epochs: arms and cycles see different
# dataset sizes, so an equal epoch cap would be an up-to-86x difference in
# gradient steps and would corrupt every compute comparison.
#
# ---------------------------------------------------------------------------
# GPU SELECTION (pass at submission; deliberately NOT a #SBATCH line here,
# because node names drift and a stale hardcoded list is worse than none)
#
# Public GPU capacity on this cluster is dominated by P100s:
#   m9g-1-[1-20], m9g-2-[1-20]   gpu:p100:4   = 160 GPUs   (features: public,gpu,pascal)
#   m13h-1-[1-2], m13h-2-[1-2]   gpu:h200:8   =  32 GPUs   (feature: hopper)
#   m13l-1-[1-2], m13l-2-[1-2]   gpu:l40s:4   =  16 GPUs   (feature: lovelace)
#   cs-*, dw-*, eng-1-1, msg-1-1  a100/h100/b200/v100 -- all tagged PRIVATE
#   mgh-1-[1-2]                  gpu:gh200:1  -- ARM, x86 conda env will not run
#   m8g-1-5                      gpu:4        -- untyped GRES, unknown hardware
#
# NOTE: m13h/m13l do NOT carry the `gpu` feature, so --constraint=gpu would
# exclude exactly the fast nodes.
#
# USE THIS -- fast nodes only, positive selection, fails loudly if unavailable:
#       --constraint='hopper|lovelace'
#
# Rationale: measured experience on this cluster is that P100s cost more wall
# time than the extra queue wait for a bigger GPU. That holds doubly for this
# job, because m9g nodes are BROADWELL-era CPUs and a meaningful part of the
# work is CPU-bound regardless of GPU -- ~150k NumPy RK4 integration steps per
# task plus 10 CPU-side SVDs for effective rank. A faster GPU cannot hide those.
#
# Fallback if m13h/m13l turn out to be unschedulable for this account
# (they carry neither a `public` nor `private` tag, so access is unconfirmed):
#       --exclude=m9g-1-[1-20],m9g-2-[1-20],mgh-1-[1-2],m8g-1-5
# That avoids Pascal, ARM (x86 env will not run), and the untyped m8g-1-5,
# while still allowing anything else the account can reach.
#
# Aside, only relevant if Pascal is ever reconsidered: CUDA 13.x dropped sm_60
# entirely, so check `python -c "import torch; print(torch.cuda.get_arch_list())"`
# before assuming a P100 would even run.
# ---------------------------------------------------------------------------

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1     # Nothing here is CPU-parallel; 1 core schedules sooner.
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-239         # PLACEHOLDER -- confirm with --print_array_specs.
#SBATCH -J l96_warmstart
#SBATCH --output=slurm_logs/warmstart_%A_%a.out
#SBATCH --error=slurm_logs/warmstart_%A_%a.err

# mkdir -p slurm_logs before submitting -- SLURM will not create it for you.

# Fail loudly rather than silently running the default preset into the wrong directory.
: "${L96_SWEEP:?not set -- submit with --export=ALL,L96_SWEEP=e1e2}"

mamba activate weight-init-effects

cd "$SLURM_SUBMIT_DIR"   # the repo root you ran sbatch from (NOT $(dirname "$0") --
                          # SLURM copies this script to /var/spool/slurmd before running it)

# #SBATCH -J is fixed at submission time, before $SLURM_ARRAY_TASK_ID exists, so
# every task would otherwise appear in squeue as an identical row. Rename at
# runtime by decoding through sweep_warmstart.decode_task_id() -- the same source
# of truth the run itself uses -- rather than duplicating the grid here.
CELL_NAME=$(python -c "
from sweep_warmstart import decode_task_id
from sweep_config import PRESET
c = decode_task_id(${SLURM_ARRAY_TASK_ID})
print(f\"{PRESET}_{c['arm']}_lr{c['lr']:g}_noise{c['noise']:g}_seed{c['seed']}\")
")
# Address the array ELEMENT as <arrayjobid>_<taskid>; $SLURM_JOB_ID can land the
# rename on a sibling task (observed in the catdd sweep).
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="$CELL_NAME" || true

python sweep_warmstart.py --task_id "$SLURM_ARRAY_TASK_ID"

# ============================================================================
# ONE-TIME SETUP (before the first submission)
#   mamba create -n weight-init-effects python=3.12 pytorch numpy matplotlib -c conda-forge
#   mkdir -p slurm_logs results
# No data transfer is needed: L96 trajectories are generated from the equations
# at runtime, so compute nodes need no outbound internet and no dataset copy.
#
# ROLLOUT
# 1. Submit the whole array in one go:
#      mkdir -p slurm_logs
#      sbatch --export=ALL,L96_SWEEP=e1e2b --constraint='hopper|lovelace' \
#             --array=0-239 slurm/warmstart_array.sbatch
#    Watch with:  squeue -u $USER
#
# 2. Aggregate into per-model-type tables once everything finishes:
#      python aggregate.py
#
# RE-RUNNING FAILURES
# Tasks are independent and each writes only its own results/<preset>/task<NNN>_*.csv,
# so a failed or PREEMPTED task can simply be resubmitted with --array=<id>.
# Writer truncates on open, so a re-run rewrites that task's CSVs cleanly -- no
# need to delete anything first, and automatic requeue under a preemptible QOS
# is safe.
# Per-cycle checkpoints land in results/<preset>/ckpt/ (warm_all only, ~4.2 GB total);
# the deferred shrink-and-perturb sweep branches from these instead of replaying chains.
# ============================================================================
