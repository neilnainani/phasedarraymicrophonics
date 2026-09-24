#######################################
#
#      Phased Array Microphonics
# Train the array-response model and compare it
# against simple interpolation baselines on
# held-out grid positions
#
# Usage (from PhasedArray_Software/):
#   python -m ai_model.prepare_dataset
#   python -m ai_model.train [--steps N]
#
#######################################

import os
import sys
import time
import argparse
import torch

from .config import Config
from .dataset import IRBandData, band_error_db, DB_PER_LOG_POWER
from .model import IRFieldNet


def baselines(data):
    """Val-set error of three non-learned predictors. The network must beat these to be worth it."""
    P, M, B, T = data.shape
    db = data.features * DB_PER_LOG_POWER
    tr, pos = data.train_idx, data.positions
    out = {"mean of train positions": [], "nearest train position": [], "linear interpolation": []}
    for v in data.val_idx:
        d = torch.cdist(pos[v].view(1, 3), pos[tr]).flatten()
        order = torch.argsort(d)
        a, b = tr[order[0]], tr[order[1]]
        wa = d[order[1]] / (d[order[0]] + d[order[1]])
        for m in range(M):
            true = db[v, m]
            out["mean of train positions"].append(band_error_db(db[tr, m].mean(0), true))
            out["nearest train position"].append(band_error_db(db[a, m], true))
            out["linear interpolation"].append(band_error_db(wa * db[a, m] + (1 - wa) * db[b, m], true))
    return {k: sum(v) / len(v) for k, v in out.items()}


@torch.no_grad()
def evaluate(model, data, split_idx):
    model.eval()
    _, M, B, T = data.shape
    scores = []
    for p in split_idx:
        for m in range(M):
            pred = data.to_db(model.predict_spectrogram(data.pos_n[p], m, B, T).cpu())
            scores.append(band_error_db(pred, data.features[p, m] * DB_PER_LOG_POWER))
    model.train()
    return sum(scores) / len(scores)


def run(cfg, log=print):
    """Train one model. Returns (best val error dB, baseline errors)."""
    tc, mc = cfg.train, cfg.model
    torch.manual_seed(tc.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = IRBandData(cfg.data.cache_file, cfg, device)
    P, M, B, T = data.shape
    log(f"Data: {P} positions ({len(data.train_idx)} train / {len(data.val_idx)} val) x {M} mics x "
        f"{B} mel bands x {T} frames; device={device}")

    model = IRFieldNet(M, mc.pos_freqs, mc.tf_freqs, mc.mic_embed_dim, mc.hidden, mc.layers, mc.skip_at).to(device)
    log(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    base = baselines(data) if len(data.val_idx) else {}
    for k, v in base.items():
        log(f"  baseline val err {k:26s} {v:6.2f} dB")

    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=tc.steps, pct_start=0.05)
    best, t0 = float("inf"), time.time()
    history = []
    for step in range(1, tc.steps + 1):
        pos, mic, ft, target = data.sample(tc.batch_points)
        loss = torch.nn.functional.l1_loss(model(pos, mic, ft), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % tc.eval_every == 0 or step == tc.steps:
            train_err = evaluate(model, data, data.train_idx)
            val_err = evaluate(model, data, data.val_idx) if len(data.val_idx) else float("nan")
            history.append({"step": step, "loss": loss.item(), "train_err": train_err, "val_err": val_err})
            mark = ""
            score = val_err if len(data.val_idx) else train_err
            if score < best:
                best, mark = score, "  *saved"
                torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                            "norm": data.normalizer_state(), "step": step,
                            "val_err": val_err, "baselines": base}, tc.checkpoint_file)
            log(f"step {step:5d}  loss {loss.item():.4f}  train err {train_err:5.2f} dB  "
                f"val err {val_err:5.2f} dB  ({time.time() - t0:5.0f}s){mark}")

    log(f"\nBest val err {best:.2f} dB  (checkpoint: {tc.checkpoint_file})")
    if base:
        b = min(base.values())
        verdict = "beats" if best < b else "does NOT beat"
        log(f"Network {verdict} the best baseline ({b:.2f} dB).")
    torch.save(history, os.path.join(os.path.dirname(tc.checkpoint_file), "train_history.pt"))
    return best, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.steps:
        cfg.train.steps = args.steps
    if not os.path.isfile(cfg.data.cache_file):
        print(f"{cfg.data.cache_file} not found - run `python -m ai_model.prepare_dataset` first.")
        return 1
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
