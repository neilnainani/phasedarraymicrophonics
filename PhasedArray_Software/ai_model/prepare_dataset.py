#######################################
#
#      Phased Array Microphonics
# Validate the grid recordings and turn them
# into the impulse-response training set
#
# Usage (from PhasedArray_Software/):
#   python -m ai_model.prepare_dataset              # validate + build cache
#   python -m ai_model.prepare_dataset --check-only # validate only
#
# Outputs (ai_model/data/):
#   manifest.csv            one row per mic file with its checks
#   validation_report.txt   human-readable summary
#   ir_dataset.pt           tensors used by train.py
#
#######################################

import os
import sys
import csv
import argparse
import numpy as np

from .config import Config
from . import acoustics as ac

SPEED_OF_SOUND = 343.0


def check_take(name, pos, deconv, cfg):
    """Run every per-file and per-take check. Returns (rows, take_info, irs_or_None)."""
    d = cfg.data
    indiv = os.path.join(d.micrecord_dir, name, "INDIV")
    rows, recs, problems, warnings = [], {}, [], []

    for mic in d.mic_ids:
        path = os.path.join(indiv, f"Mic{mic}_{name}.wav")
        row = {"take": name, "x": pos[0], "y": pos[1], "z": pos[2], "mic": mic,
               "path": os.path.relpath(path, d.micrecord_dir)}
        if not os.path.isfile(path):
            row["status"] = "MISSING"
            problems.append(f"Mic{mic} missing")
            rows.append(row)
            continue
        sr, x, info = ac.read_wav(path)
        row.update(sr=sr, dtype=info["dtype"], channels=info["channels"],
                   duration_s=round(len(x) / sr, 4),
                   peak=round(float(np.max(np.abs(x))), 4),
                   rms_db=round(float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-20)), 2),
                   dc=float(np.mean(x)),
                   hum120_db=round(ac.hum_level_db(x, sr), 2))
        if sr != d.capture_sr:
            problems.append(f"Mic{mic} sample rate {sr} != {d.capture_sr}")
        if info["channels"] != 1:
            problems.append(f"Mic{mic} is not mono")
        if row["peak"] >= d.clip_level:
            warnings.append(f"Mic{mic} clips (peak {row['peak']})")
        if abs(row["duration_s"] - d.expected_duration_s) > 0.05:
            warnings.append(f"Mic{mic} duration {row['duration_s']} s != {d.expected_duration_s} s")
        recs[mic] = x
        rows.append(row)

    extra = sorted(f for f in os.listdir(indiv) if f.endswith(".wav")) if os.path.isdir(indiv) else []
    unexpected = [f for f in extra if not any(f == f"Mic{m}_{name}.wav" for m in d.mic_ids)]
    if unexpected:
        warnings.append(f"unexpected files ignored: {unexpected}")

    take = {"take": name, "x": pos[0], "y": pos[1], "z": pos[2]}
    if len(recs) != len(d.mic_ids) or problems:
        take.update(status="FAIL", problems=problems, warnings=warnings)
        return rows, take, None

    lengths = {len(x) for x in recs.values()}
    if len(lengths) != 1:
        warnings.append(f"mic files have different lengths {sorted(lengths)}; trimming to shortest")
    n = min(lengths)
    irs = {m: deconv(recs[m][:n]) for m in d.mic_ids}
    onset, _ = ac.take_onset(list(irs.values()), d.capture_sr, d)
    onset_s = ac.signed_lag(onset, deconv.n) / d.capture_sr

    # Did the part of the sweep we train on land inside the recording window?
    t_lo = onset_s + ac.ess_time_of_freq(d.band_lo_hz, d)
    t_hi = onset_s + ac.ess_time_of_freq(d.band_hi_hz, d)
    if t_lo < 0 or t_hi > n / d.capture_sr:
        problems.append(f"sweep band {d.band_lo_hz:.0f}-{d.band_hi_hz:.0f} Hz falls outside the "
                        f"recording (needs {t_lo:.2f}..{t_hi:.2f} s of a {n / d.capture_sr:.2f} s file)")
    lost_below_hz = d.sweep_f1 * (d.sweep_f2 / d.sweep_f1) ** (max(0.0, -onset_s) / d.sweep_duration_s)
    if onset_s < 0 and lost_below_hz >= d.band_lo_hz / 2:
        warnings.append(f"playback started {-onset_s * 1000:.0f} ms before recording "
                        f"(sweep content below {lost_below_hz:.0f} Hz lost)")

    pnr = {}
    for m in d.mic_ids:
        pnr[m] = ac.peak_to_noise_db(irs[m], onset, d.capture_sr)
        for r in rows:
            if r["mic"] == m:
                r["ir_pnr_db"] = round(pnr[m], 1)
        if pnr[m] < d.min_ir_pnr_db:
            problems.append(f"Mic{m} impulse response only {pnr[m]:.1f} dB above noise "
                            f"(< {d.min_ir_pnr_db} dB) - sweep not detected")

    take.update(onset_ms=round(onset_s * 1000, 1), lost_below_hz=round(lost_below_hz, 1), ir_pnr_db={m: round(v, 1) for m, v in pnr.items()},
                status="FAIL" if problems else ("WARN" if warnings else "OK"),
                problems=problems, warnings=warnings)
    for r in rows:
        r["status"] = take["status"]

    pre = int(d.pre_roll_ms * 1e-3 * d.capture_sr)
    length = int(d.ir_length_s * d.capture_sr)
    cropped = np.stack([ac.crop_circular(irs[m], onset - pre, length) for m in d.mic_ids])
    return rows, take, cropped


def grid_summary(takes, cfg):
    ok = [t for t in takes if t["status"] != "FAIL"]
    pos = np.array([[t["x"], t["y"], t["z"]] for t in ok]) if ok else np.zeros((0, 3))
    lines = [f"Usable grid points: {len(ok)} / {len(takes)}"]
    varying = []
    for i, axis in enumerate("XYZ"):
        vals = np.unique(np.round(pos[:, i], 4)) if len(pos) else []
        if len(vals) <= 1:
            lines.append(f"  {axis}: constant {vals[0] if len(vals) else '-'}  (model cannot learn this axis)")
            continue
        steps = np.diff(vals)
        varying.append(axis)
        lines.append(f"  {axis}: {len(vals)} values {vals.min()}..{vals.max()} m, spacing "
                     f"{steps.min():.3f}-{steps.max():.3f} m")
        missing = np.round(np.arange(vals.min(), vals.max() + 1e-9, steps.min()), 4)
        gaps = sorted(set(missing) - set(np.round(vals, 4)))
        if gaps:
            lines.append(f"     gaps at {gaps}")
        f_alias = SPEED_OF_SOUND / (2 * steps.min())
        lines.append(f"     spatial Nyquist: phase is only sampled finely enough below ~{f_alias:.0f} Hz; "
                     f"above that the model can learn level/reverb, not exact waveforms")
    lines.append(f"  Grid dimensionality: {len(varying)}-D ({', '.join(varying) or 'none'})")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    d = cfg.data
    os.makedirs(d.report_dir, exist_ok=True)

    sr, dry, _ = ac.read_wav(os.path.join(d.micrecord_dir, d.dry_sweep_file))
    assert sr == d.capture_sr, f"dry sweep is {sr} Hz, expected {d.capture_sr}"
    points = ac.list_grid_points(d.micrecord_dir, d.grid_prefix)
    bad_names = [n for n, p in points if p is None]
    points = [(n, p) for n, p in points if p is not None]
    others = [n for n in os.listdir(d.micrecord_dir)
              if os.path.isdir(os.path.join(d.micrecord_dir, n)) and not n.startswith(d.grid_prefix)]
    deconv = ac.Deconvolver(dry, sr, int(d.expected_duration_s * sr) + sr, d)

    all_rows, takes, irs, kept = [], [], [], []
    for name, pos in points:
        rows, take, cropped = check_take(name, pos, deconv, cfg)
        all_rows += rows
        takes.append(take)
        flag = {"OK": "  ok ", "WARN": " warn", "FAIL": " FAIL"}[take["status"]]
        detail = "; ".join(take["problems"] + take["warnings"])
        print(f"[{flag}] {name:28s} onset {take.get('onset_ms', float('nan')):8.1f} ms  {detail}")
        if cropped is not None and take["status"] != "FAIL":
            irs.append(cropped)
            kept.append(take)

    # Per-channel gain: median level of each mic across takes (channels are not gain-matched).
    gains = {m: np.median([r["rms_db"] for r in all_rows if r["mic"] == m and "rms_db" in r])
             for m in d.mic_ids}

    manifest = os.path.join(d.report_dir, "manifest.csv")
    keys = ["take", "x", "y", "z", "mic", "path", "status", "sr", "dtype", "channels", "duration_s",
            "peak", "rms_db", "dc", "hum120_db", "ir_pnr_db"]
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    lines = ["Phased Array grid dataset - validation report", "=" * 46, ""]
    lines += grid_summary(takes, cfg)
    lines += ["", "Channel levels (median RMS incl. noise): " +
              ", ".join(f"Mic{m} {g:.1f} dB" for m, g in gains.items())]
    if max(gains.values()) - min(gains.values()) > 3:
        lines.append("  -> channels differ by >3 dB: preamp gains are not matched. The model learns each"
                     " mic's gain as-is; calibrate before comparing mics physically.")
    early = [t for t in takes if t.get("onset_ms", 0) < 0]
    if early:
        worst = min(early, key=lambda t: t["onset_ms"])
        lines.append(f"\nPlayback/recording are not synchronized: in {len(early)} takes the sweep started before "
                     f"recording (worst {-worst['onset_ms']:.0f} ms, losing only < {worst['lost_below_hz']:.0f} Hz)."
                     "\n  Absolute arrival time is therefore unknown; only timing *between* mics of the same take is"
                     "\n  kept. Starting playback ~0.5 s after recording starts (capture.sh) would remove this.")
    if bad_names:
        lines.append(f"\nFolders starting with '{d.grid_prefix}' whose name could not be parsed: {bad_names}")
    if others:
        lines.append(f"\nNon-grid sessions ignored (no position/sweep): {sorted(others)}")
    lines += ["", "Per-take results:"]
    for t in takes:
        lines.append(f"  [{t['status']}] {t['take']}  onset={t.get('onset_ms')} ms  PNR={t.get('ir_pnr_db')}")
        for p in t["problems"]:
            lines.append(f"      problem: {p}")
        for wmsg in t["warnings"]:
            lines.append(f"      warning: {wmsg}")
    report = "\n".join(lines)
    with open(os.path.join(d.report_dir, "validation_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\nWrote {manifest}")

    if args.check_only:
        return 0 if kept else 1
    if not kept:
        print("No usable takes - nothing to cache.")
        return 1

    import torch
    m = cfg.model
    irs_model = np.stack([[ac.to_model_rate(h, d.capture_sr, m.model_sr) for h in take] for take in irs])
    torch.save({
        "positions": torch.tensor([[t["x"], t["y"], t["z"]] for t in kept], dtype=torch.float32),
        "mic_ids": list(d.mic_ids),
        "irs": torch.tensor(irs_model, dtype=torch.float32),   # [P, M, L] at model_sr
        "sr": m.model_sr,
        "takes": [t["take"] for t in kept],
        "onset_ms": [t["onset_ms"] for t in kept],
        "config": cfg.to_dict(),
    }, d.cache_file)
    print(f"Wrote {d.cache_file}: {irs_model.shape[0]} positions x {irs_model.shape[1]} mics x "
          f"{irs_model.shape[2]} samples @ {m.model_sr} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
