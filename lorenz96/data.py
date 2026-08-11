import numpy as np
import torch

from .system import DT, K, f_schedule, initial_state, integrate

REFERENCE_F = 8.0
SPINUP = 5_000

_traj_cache = {}
_norm_cache = {}


def load_trajectory(F=REFERENCE_F, n_steps=100_000, seed=0, dt=DT, spinup=SPINUP):
    key = (float(F) if np.isscalar(F) else None, n_steps, seed, dt, spinup)
    if key[0] is None:
        return integrate(initial_state(REFERENCE_F, seed), F, n_steps, dt, spinup)
    if key not in _traj_cache:
        x0 = initial_state(F, seed)
        _traj_cache[key] = integrate(x0, F, n_steps, dt, spinup)
    return _traj_cache[key]


def normalizer(n_steps=100_000, seed=0):
    # Frozen at REFERENCE_F: Var(X) grows with F, so per-regime stats would fake a shift effect.
    key = (n_steps, seed)
    if key not in _norm_cache:
        traj = load_trajectory(REFERENCE_F, n_steps, seed)
        _norm_cache[key] = (float(traj.mean()), float(traj.std()))
    return _norm_cache[key]


def make_dataset(F=REFERENCE_F, n=10_000, seed=0, dt=DT, spinup=SPINUP, normalize=True):
    # Targets are tendencies X(t+dt) - X(t), not raw next states.
    traj = load_trajectory(F, n + 1, seed, dt, spinup)
    X, Y = traj[:-1], traj[1:] - traj[:-1]

    if normalize:
        mean, std = normalizer()
        X = (X - mean) / std
        Y = Y / std

    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


def load_subset(n, seed, F=REFERENCE_F, pool=100_000, **kwargs):
    X, Y = make_dataset(F, pool, seed=0, **kwargs)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)[:n]
    return X[idx], Y[idx]


def make_shifted_dataset(kind, n, F0=8.0, F1=16.0, width=0, n_cycles=2, seed=0, dt=DT,
                         spinup=SPINUP, normalize=True):
    # Non-stationary stream: F varies across the trajectory per f_schedule.
    F_seq = np.concatenate(
        [np.full(spinup, F0), f_schedule(kind, n + 1, F0, F1, width, n_cycles)])
    traj = integrate(initial_state(F0, seed), F_seq, n + 1, dt, spinup)
    X, Y = traj[:-1], traj[1:] - traj[:-1]

    if normalize:
        mean, std = normalizer()
        X = (X - mean) / std
        Y = Y / std

    return torch.from_numpy(X).float(), torch.from_numpy(Y).float(), F_seq[spinup:spinup + n]


def rollout_truth(F=REFERENCE_F, n_steps=1_000, seed=0, dt=DT, spinup=SPINUP):
    return load_trajectory(F, n_steps, seed, dt, spinup)
