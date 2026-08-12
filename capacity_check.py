import argparse
import csv
import time

import torch
import torch.nn as nn

from lorenz96.models import CircularCNN, count_params
from lorenz96.stream import CycleStream

# Val MSE against noisy targets cannot beat 2*sigma^2; train MSE below it means memorization.
def noise_floor(sigma):
    return 2.0 * sigma ** 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--noise", type=float, default=0.10)
    p.add_argument("--chunk", type=int, default=2_000)
    p.add_argument("--epochs", type=int, default=1_500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--out", default="results/capacity_check.csv")
    args = p.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    stream = CycleStream(seed=0, noise=args.noise, n_cycles=1, chunk=args.chunk,
                         n_val=4_000, history=4)
    d = stream.cycle_data(0, data_all=True)
    x, y = d.x.to(dev), d.y.to(dev)
    xv, yv, yvc = stream.val.x.to(dev), stream.val.y.to(dev), stream.val.y_clean.to(dev)

    model = CircularCNN(hidden=args.hidden, n_layers=3, kernel=5, in_channels=4).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(0)
    n, floor = x.shape[0], noise_floor(args.noise)
    steps_per_epoch = n // args.batch

    print(f"params={count_params(model):,}  patches={n * 40:,}  "
          f"ratio={count_params(model) / (n * 40):.1f}x  device={dev.type}")
    print(f"steps/epoch={steps_per_epoch}  total_steps={steps_per_epoch * args.epochs:,}  "
          f"noise floor (2*sigma^2)={floor:.4f}\n")

    rows, t0, crossed = [], time.time(), None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        run, nb = 0.0, 0
        for i in range(0, n - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            loss = nn.functional.mse_loss(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss.detach())
            nb += 1
        train_loss = run / nb

        model.eval()
        with torch.no_grad():
            pred = model(xv)
            vn = float(nn.functional.mse_loss(pred, yv))
            vc = float(nn.functional.mse_loss(pred, yvc))

        step = (epoch + 1) * steps_per_epoch
        rows.append(dict(epoch=epoch, step=step, train_loss=train_loss, val_mse_noisy=vn,
                         val_mse_clean=vc, gap=vn - train_loss, noise_floor=floor,
                         wall_s=round(time.time() - t0, 2)))
        if crossed is None and train_loss < floor:
            crossed = (epoch, step)

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:5d} step {step:6d}  train {train_loss:.5f}  "
                  f"val_noisy {vn:.5f}  gap {vn - train_loss:+.5f}  "
                  f"val_clean {vc:.5f}  [{time.time() - t0:.0f}s]", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    best = min(rows, key=lambda r: r["val_mse_clean"])
    print(f"\ntrain loss crossed noise floor at: "
          f"{'epoch %d, step %d' % crossed if crossed else 'NEVER -- no memorization'}")
    print(f"best val_mse_clean {best['val_mse_clean']:.5f} at epoch {best['epoch']} "
          f"(step {best['step']})")
    print(f"final gap {rows[-1]['gap']:+.5f}   -> {args.out}")


if __name__ == "__main__":
    main()
