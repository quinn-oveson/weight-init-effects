import argparse
import time

import torch
import torch.nn as nn

from lorenz96 import data as D
from lorenz96 import metrics as M
from lorenz96.models import CircularCNN, Persistence, count_params


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def stack_history(x, n_history):
    # Turn a (T, K) sequence into (T-n+1, n, K) sliding windows of past observations.
    if n_history == 1:
        return x
    return torch.stack([x[i:len(x) - n_history + 1 + i] for i in range(n_history)], dim=1)


def make_split(F, n_train, n_val, noise, seed, n_history):
    train = D.make_dataset(F, n_train + n_history, seed=seed, noise=noise)
    val = D.make_dataset(F, n_val + n_history, seed=seed + 1000, noise=noise)

    def prep(b):
        x = stack_history(b.x, n_history)
        y = b.y[n_history - 1:]
        y_clean = b.y_clean[n_history - 1:]
        return x, y, y_clean

    return prep(train), prep(val)


def gaussian_nll(mean, log_var, target):
    return (0.5 * (log_var + (target - mean) ** 2 / log_var.exp())).mean()


def train(model, train_set, val_set, epochs=40, batch_size=256, lr=1e-3, dev=None):
    dev = dev or device()
    model = model.to(dev)
    x, y, _ = (t.to(dev) for t in train_set)
    xv, yv, yv_clean = (t.to(dev) for t in val_set)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hetero = model.heteroscedastic

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x), device=dev)
        for i in range(0, len(x), batch_size):
            idx = perm[i:i + batch_size]
            pred = model(x[idx])
            loss = gaussian_nll(*pred, y[idx]) if hetero else nn.functional.mse_loss(pred, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred = model(xv)
        mean = pred[0] if hetero else pred
        return dict(val_mse_noisy=float(nn.functional.mse_loss(mean, yv)),
                    val_mse_clean=float(nn.functional.mse_loss(mean, yv_clean)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--hetero", action="store_true")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--F", type=float, default=8.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    train_set, val_set = make_split(args.F, args.n_train, 4_000, args.noise, args.seed,
                                    args.history)

    model = CircularCNN(hidden=args.hidden, n_layers=args.layers, in_channels=args.history,
                        heteroscedastic=args.hetero)
    t0 = time.time()
    losses = train(model, train_set, val_set, epochs=args.epochs)
    elapsed = time.time() - t0

    dev = device()
    res = M.evaluate(model.to(dev), F=args.F, history=args.history, sample=args.hetero,
                     init_noise=args.noise)
    base = M.evaluate(Persistence().to(dev), F=args.F, history=args.history,
                      init_noise=args.noise)

    print(f"noise={args.noise}  history={args.history}  hetero={args.hetero}  "
          f"params={count_params(model)}  [{elapsed:.0f}s on {dev.type}]")
    print(f"  val MSE vs noisy target : {losses['val_mse_noisy']:.5f}")
    print(f"  val MSE vs clean target : {losses['val_mse_clean']:.5f}")
    print(f"  valid time  model       : {res['steps']:4d} steps  "
          f"{res['lyapunov_times']:.2f} Lyapunov times   stable={res['stable']}")
    print(f"  valid time  persistence : {base['steps']:4d} steps  "
          f"{base['lyapunov_times']:.2f} Lyapunov times")
    print(f"  rollout std {res['pred_std']:.3f} vs truth {res['truth_std']:.3f}")


if __name__ == "__main__":
    main()
