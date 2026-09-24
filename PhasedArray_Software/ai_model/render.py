#######################################
#
#      Phased Array Microphonics
# Predict what each array mic would record if the
# array were at (x, y, z), for any dry source audio
#
# Usage (from PhasedArray_Software/):
#   python -m ai_model.render --pos 1.15 1.25 0.87 --source ../MICRECORD/dry_sweep.wav --out pred/
#
# The model predicts each mic's band-energy envelope; an impulse
# response with that envelope is synthesized from shaped noise
# (the fine structure of a reverberant IR is random anyway) and
# convolved with the source.
#
#######################################

import os
import sys
import argparse
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import fftconvolve

from .config import Config, DataConfig, ModelConfig
from .model import IRFieldNet
from .dataset import PositionNormalizer, ir_features, mel_filterbank
from . import acoustics as ac

SYNTH_N_FFT, SYNTH_HOP = 512, 128   # fine grid the envelope is interpolated onto


def load(checkpoint):
    ck = torch.load(checkpoint, weights_only=False, map_location="cpu")
    mc = ModelConfig(**ck["config"]["model"])
    norm = ck["norm"]
    model = IRFieldNet(len(norm["mic_ids"]), mc.pos_freqs, mc.tf_freqs, mc.mic_embed_dim,
                       mc.hidden, mc.layers, mc.skip_at)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, mc, DataConfig(**ck["config"]["data"]), norm


def predict_features(model, mc, norm, pos_m):
    """[M, n_mels, T] predicted log band power for every mic at `pos_m` (metres)."""
    pos_norm = PositionNormalizer(norm["pos"]["center"], norm["pos"]["half"])
    _, M, B, T = norm["shape"]
    p = pos_norm(torch.tensor(pos_m, dtype=torch.float32))
    return torch.stack([model.predict_spectrogram(p, m, B, T) * norm["std"] + norm["mean"] for m in range(M)])


def synthesize_ir(log_band_power, mc, dc, seed=0, iters=8):
    """Shaped-noise impulse response whose band envelope matches `log_band_power` [n_mels, T]."""
    length = int(round(dc.ir_length_s * mc.model_sr))
    fine_mc = ModelConfig(**{**mc.__dict__, "n_fft": SYNTH_N_FFT, "hop": SYNTH_HOP})
    fb = mel_filterbank(fine_mc, dc)                                    # [F, B]
    spread = fb / fb.sum(1, keepdim=True).clamp(min=1e-12)             # band -> linear-bin weights
    in_band = (fb.sum(1) > 0).float().unsqueeze(1)

    B, T = log_band_power.shape
    coarse_t = torch.arange(T) * mc.hop
    fine_t = torch.arange(length // SYNTH_HOP + 1) * SYNTH_HOP
    gain = torch.zeros(B, T)                                            # per-cell correction, log power
    window = torch.hann_window(SYNTH_N_FFT)
    noise = torch.from_numpy(np.random.default_rng(seed).standard_normal(length)).float()
    N = torch.stft(noise, SYNTH_N_FFT, SYNTH_HOP, window=window, return_complex=True)
    N = N / N.abs().clamp(min=1e-12)                                    # unit-magnitude random phase

    for _ in range(iters):
        target = log_band_power + gain
        # interpolate each band's log power from the coarse frame times onto the fine frames
        idx = torch.searchsorted(coarse_t, fine_t).clamp(1, T - 1)
        w = ((fine_t - coarse_t[idx - 1]) / (coarse_t[idx] - coarse_t[idx - 1])).clamp(0, 1)
        fine_log = target[:, idx - 1] * (1 - w) + target[:, idx] * w          # [B, T_fine]
        bin_power = (spread @ torch.exp(fine_log)) * in_band                   # [F, T_fine]
        ir = torch.istft(N * bin_power.sqrt(), SYNTH_N_FFT, SYNTH_HOP, window=window, length=length)
        err = log_band_power - ir_features(ir, mc, dc)                         # [B, T]
        gain = gain + err
    return ir.numpy(), float(err.abs().mean() * 10 / np.log(10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--source", required=True, help="dry mono wav to play through the room")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default=Config().train.checkpoint_file)
    args = parser.parse_args()

    model, mc, dc, norm = load(args.checkpoint)
    tp = norm["train_positions"]
    lo, hi = tp.min(0).values, tp.max(0).values
    outside = [a for i, a in enumerate("XYZ") if not (lo[i] - 1e-6 <= args.pos[i] <= hi[i] + 1e-6)]
    if outside:
        print(f"WARNING: {', '.join(outside)} outside the trained range "
              f"{lo.tolist()} .. {hi.tolist()} - this is extrapolation and will not be accurate.")

    feats = predict_features(model, mc, norm, args.pos)
    sr, src, _ = ac.read_wav(args.source)
    if src.ndim > 1:
        src = src.mean(1)
    src = ac.to_model_rate(src, sr, mc.model_sr)

    os.makedirs(args.out, exist_ok=True)
    tag = f"X{args.pos[0]:g}_Y{args.pos[1]:g}_Z{args.pos[2]:g}"
    outs = []
    for m, mic in enumerate(norm["mic_ids"]):
        ir, fit_db = synthesize_ir(feats[m], mc, dc, seed=m)
        np.save(os.path.join(args.out, f"Mic{mic}_PredIR_{tag}.npy"), ir)
        outs.append(fftconvolve(src, ir))
        print(f"Mic{mic}: synthesized IR matches predicted envelope to {fit_db:.2f} dB")
    # One shared scale so the level differences between mics are preserved.
    peak = max(np.max(np.abs(o)) for o in outs) + 1e-12
    for mic, o in zip(norm["mic_ids"], outs):
        wavfile.write(os.path.join(args.out, f"Mic{mic}_Pred_{tag}.wav"), mc.model_sr,
                      (0.9 * o / peak).astype(np.float32))
    print(f"Wrote {len(outs)} predicted mic signals to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
