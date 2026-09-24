#######################################
#
#      Phased Array Microphonics
# Signal-processing helpers for turning sweep
# recordings into room impulse responses
#
#######################################

import os
import re
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

GRID_RE = re.compile(r"^Grid_X(-?[\d.]+)_Y(-?[\d.]+)_Z(-?[\d.]+)$")


def parse_grid_name(name):
    """'Grid_X1.15_Y0.3_Z0.87' -> (1.15, 0.3, 0.87), or None if the folder is not a grid point."""
    m = GRID_RE.match(name)
    return tuple(float(v) for v in m.groups()) if m else None


def read_wav(path):
    """Read a wav as float32 in [-1, 1]. Returns (sr, x, info)."""
    sr, x = wavfile.read(path)
    info = {"dtype": str(x.dtype), "channels": 1 if x.ndim == 1 else x.shape[1]}
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128.0) / 128.0
    else:
        x = x.astype(np.float32)
    return sr, x, info


def ess_time_of_freq(f, cfg):
    """Time (s) after sweep start at which the exponential sweep passes frequency f."""
    return cfg.sweep_duration_s * np.log(f / cfg.sweep_f1) / np.log(cfg.sweep_f2 / cfg.sweep_f1)


def _band_window(freqs, lo, hi, taper=0.25):
    """Flat-top band-pass weights with raised-cosine edges (taper = fraction of an octave-ish)."""
    w = np.zeros_like(freqs)
    lo0, hi1 = lo * (1 - taper), hi * (1 + taper)
    w[(freqs >= lo) & (freqs <= hi)] = 1.0
    rise = (freqs > lo0) & (freqs < lo)
    w[rise] = 0.5 - 0.5 * np.cos(np.pi * (freqs[rise] - lo0) / (lo - lo0))
    fall = (freqs > hi) & (freqs < hi1)
    w[fall] = 0.5 + 0.5 * np.cos(np.pi * (freqs[fall] - hi) / (hi1 - hi))
    return w


class Deconvolver:
    """Regularized spectral division by the dry sweep. The result is circular, so a take whose
    playback started before the recording has its impulse at a negative lag (end of the array)."""

    def __init__(self, dry, sr, rec_len, cfg):
        self.n = 1 << int(np.ceil(np.log2(rec_len + len(dry))))
        self.sr = sr
        X = np.fft.rfft(dry, self.n)
        power = np.abs(X) ** 2
        freqs = np.fft.rfftfreq(self.n, 1.0 / sr)
        self.inv = np.conj(X) / (power + cfg.deconv_regularization * power.max())
        self.inv *= _band_window(freqs, cfg.band_lo_hz, cfg.band_hi_hz)

    def __call__(self, rec):
        return np.fft.irfft(np.fft.rfft(rec, self.n) * self.inv, self.n)


def take_onset(irs, sr, cfg):
    """Common onset (circular index) for all mics of one take, so relative mic timing is kept."""
    env = np.sum([h ** 2 for h in irs], axis=0)
    k = max(1, int(0.0005 * sr))
    env = np.convolve(np.concatenate([env[-k:], env]), np.ones(k) / k, mode="valid")[1:]
    peak = int(np.argmax(env))
    thr = env[peak] * 10 ** (-cfg.onset_threshold_db / 10)
    search = int(cfg.onset_search_ms * 1e-3 * sr)
    idx = (peak - np.arange(search, -1, -1)) % len(env)
    above = np.nonzero(env[idx] >= thr)[0]
    return int(idx[above[0]]), peak


def signed_lag(idx, n):
    return idx - n if idx > n // 2 else idx


def crop_circular(h, start, length):
    return h[(start + np.arange(length)) % len(h)]


def peak_to_noise_db(h, onset, sr):
    """Direct/early peak vs. the noise-only region 0.8-1.3 s after the onset."""
    seg = crop_circular(h, onset, int(0.1 * sr))
    noise = crop_circular(h, onset + int(0.8 * sr), int(0.5 * sr))
    return float(20 * np.log10(np.max(np.abs(seg)) / (np.sqrt(np.mean(noise ** 2)) + 1e-20)))


def hum_level_db(x, sr, f0=120.0):
    """Level of the mains-hum harmonic relative to the full-band RMS."""
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    band = (freqs > f0 - 3) & (freqs < f0 + 3)
    return float(10 * np.log10(np.sum(X[band] ** 2) / np.sum(X ** 2) + 1e-20))


def to_model_rate(h, src_sr, dst_sr):
    g = np.gcd(src_sr, dst_sr)
    return resample_poly(h, dst_sr // g, src_sr // g)


def list_grid_points(micrecord_dir, prefix="Grid_"):
    points = []
    for name in sorted(os.listdir(micrecord_dir)):
        if not name.startswith(prefix):
            continue
        pos = parse_grid_name(name)
        points.append((name, pos))
    return points
