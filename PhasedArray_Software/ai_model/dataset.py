#######################################
#
#      Phased Array Microphonics
# Training data for the array-response model:
# mel-band energy envelopes of the extracted
# impulse responses, split by grid position
#
# Why bands and not raw STFT bins: in this room the
# per-bin magnitude of an impulse response changes
# randomly within 0.1 m (diffuse-field fluctuations,
# ~5.6 dB spread), so raw bins are not predictable from
# position - nearest-neighbour did worse than the mean.
# Band energy over time (level, colouration, decay) is
# smooth in space and is what a listener hears.
#
#######################################

import math
import torch
import torchaudio

DB_PER_LOG_POWER = 10.0 / math.log(10.0)


def mel_filterbank(mcfg, dcfg):
    """[F, n_mels] triangular weights covering the usable band."""
    return torchaudio.functional.melscale_fbanks(
        n_freqs=mcfg.n_fft // 2 + 1, f_min=dcfg.band_lo_hz, f_max=dcfg.band_hi_hz,
        n_mels=mcfg.n_mels, sample_rate=mcfg.model_sr, norm=None, mel_scale="htk")


def ir_features(irs, mcfg, dcfg):
    """[..., L] impulse responses -> [..., n_mels, T] log of the mean power in each mel band."""
    shape = irs.shape[:-1]
    spec = torch.stft(irs.reshape(-1, irs.shape[-1]), n_fft=mcfg.n_fft, hop_length=mcfg.hop,
                      window=torch.hann_window(mcfg.n_fft), center=True, return_complex=True)
    fb = mel_filterbank(mcfg, dcfg)
    band_power = torch.einsum("bft,fm->bmt", spec.abs() ** 2, fb) / fb.sum(0).view(1, -1, 1)
    return torch.log(band_power + mcfg.log_eps).reshape(*shape, mcfg.n_mels, spec.shape[-1])


class PositionNormalizer:
    """Maps metres to ~[-1, 1] per axis; constant axes (X, Z today) map to 0."""

    def __init__(self, center, half):
        self.center, self.half = center, half

    @classmethod
    def fit(cls, positions, min_half):
        lo, hi = positions.min(0).values, positions.max(0).values
        return cls((lo + hi) / 2, torch.clamp((hi - lo) / 2, min=min_half))

    def __call__(self, pos):
        return (pos - self.center.to(pos.device)) / self.half.to(pos.device)

    def state(self):
        return {"center": self.center, "half": self.half}


class IRBandData:
    def __init__(self, cache_file, cfg, device="cpu"):
        blob = torch.load(cache_file, weights_only=False)
        self.cfg = cfg
        self.mic_ids = blob["mic_ids"]
        self.positions = blob["positions"]                               # [P, 3] metres
        self.features = ir_features(blob["irs"], cfg.model, cfg.data)   # [P, M, B, T] log power
        P, M, B, T = self.features.shape
        self.shape = (P, M, B, T)

        val = torch.tensor(cfg.train.val_positions, dtype=torch.float32)
        is_val = (torch.cdist(self.positions, val) < 1e-3).any(1) if len(val) else torch.zeros(P, dtype=torch.bool)
        self.val_idx = torch.nonzero(is_val).flatten()
        self.train_idx = torch.nonzero(~is_val).flatten()
        if len(self.train_idx) == 0:
            raise ValueError("every position is in the validation set")

        self.pos_norm = PositionNormalizer.fit(self.positions[self.train_idx], cfg.model.min_pos_half_range_m)
        train = self.features[self.train_idx]
        self.mean, self.std = train.mean().item(), train.std().item()

        self.device = device
        self.target = ((self.features - self.mean) / self.std).to(device)
        self.pos_n = self.pos_norm(self.positions).to(device)
        self.b_n = torch.linspace(-1, 1, B, device=device)
        self.t_n = torch.linspace(-1, 1, T, device=device)

    def sample(self, n, split="train"):
        """Random (position, mic, band, time) queries with their targets."""
        idx = (self.train_idx if split == "train" else self.val_idx).to(self.device)
        _, M, B, T = self.shape
        p = idx[torch.randint(len(idx), (n,), device=self.device)]
        m = torch.randint(M, (n,), device=self.device)
        b = torch.randint(B, (n,), device=self.device)
        t = torch.randint(T, (n,), device=self.device)
        ft = torch.stack([self.b_n[b], self.t_n[t]], dim=-1)
        return self.pos_n[p], m, ft, self.target[p, m, b, t]

    def to_db(self, normalized):
        return (normalized * self.std + self.mean) * DB_PER_LOG_POWER

    def normalizer_state(self):
        return {"pos": self.pos_norm.state(), "mean": self.mean, "std": self.std,
                "shape": self.shape, "mic_ids": self.mic_ids,
                "train_positions": self.positions[self.train_idx]}


def band_error_db(pred_db, true_db, floor_db=60.0):
    """RMS error (dB) over [band, time] cells within `floor_db` of the IR's loudest cell,
    so the score is not dominated by matching the noise floor after the decay."""
    mask = true_db > (true_db.max() - floor_db)
    return torch.sqrt(((pred_db - true_db) ** 2)[mask].mean()).item()
