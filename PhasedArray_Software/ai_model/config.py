#######################################
#
#      Phased Array Microphonics
# Configuration for the array-response model
#
# Every number here was chosen from an analysis of the
# Grid_X*_Y*_Z* sweep recordings (Fall 2026); the reason
# for each value is written next to it.
#
#######################################

import os
from dataclasses import dataclass, field, asdict

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DataConfig:
    # --- Where things live ---
    micrecord_dir: str = os.path.normpath(os.path.join(_HERE, "..", "..", "MICRECORD"))
    dry_sweep_file: str = "dry_sweep.wav"
    grid_prefix: str = "Grid_"
    cache_file: str = os.path.join(_HERE, "data", "ir_dataset.pt")
    report_dir: str = os.path.join(_HERE, "data")

    # --- Capture format (what collect_dataset_point.py writes) ---
    capture_sr: int = 44100
    mic_ids: tuple = (2, 3, 4)          # Mic1 is intentionally not part of the grid dataset
    expected_duration_s: float = 5.0

    # --- Excitation: exponential sine sweep in dry_sweep.wav (measured 20 Hz -> 20 kHz in 3.0 s) ---
    sweep_f1: float = 20.0
    sweep_f2: float = 20000.0
    sweep_duration_s: float = 3.0

    # --- Usable band ---
    # Measured impulse-response dynamic range per octave: 125 Hz 1 dB, 250 Hz 4 dB, 500 Hz 17 dB,
    # 1 kHz 28 dB, 2-8 kHz 42-54 dB. Below ~500 Hz the sweep is buried under hum/noise; above
    # ~9-10 kHz the recordings contain no sweep energy. Everything outside this band is noise.
    band_lo_hz: float = 500.0
    band_hi_hz: float = 7500.0

    # --- Impulse-response extraction ---
    deconv_regularization: float = 1e-4  # Tikhonov term, relative to peak |X(f)|^2 of the sweep
    onset_threshold_db: float = 20.0     # take onset = first point within 20 dB of the envelope peak
    onset_search_ms: float = 50.0
    pre_roll_ms: float = 5.0             # keep 5 ms before the onset so no mic's direct sound is cut
    ir_length_s: float = 0.35            # decays ~20 dB per 125 ms (RT60 ~0.38 s above 1 kHz) and
                                         # reaches the ~55 dB noise floor by ~0.35 s

    # --- Quality gates (validate_dataset.py) ---
    min_ir_pnr_db: float = 30.0          # all current takes measure ~40-60 dB
    clip_level: float = 0.999


@dataclass
class ModelConfig:
    model_sr: int = 16000       # speech application + no energy above ~10 kHz in the data
    # Target = log energy in 16 mel bands (500-7500 Hz) x 64 ms frames (6 frames per IR).
    # Finer targets were measured to be unpredictable from position with this grid: raw 31 Hz x 8 ms
    # STFT bins and 32 bands x 8 ms both vary randomly between points 0.1 m apart (interpolating
    # neighbours did worse than the global mean). At 8-16 bands x 64 ms, interpolation starts to
    # beat the mean, i.e. there is real spatial signal to learn.
    n_fft: int = 2048           # 128 ms window
    hop: int = 1024             # 64 ms hop
    log_eps: float = 1e-10
    n_mels: int = 16

    # Positional encoding / network size, chosen by a sweep on held-out positions (5 seeds each):
    #   pos_freqs 0/1/2/3/4 -> 1.030/1.011/1.013/1.040/1.11 dB; smooth spatial prior wins.
    #   hidden x layers 64x3 / 128x4 / 256x6 -> 1.05 / 1.02 / 1.01 dB; weight decay had no effect.
    # 128x4 is as good as the larger net at ~1/7 the parameters. Revisit when the grid is 2-D/3-D.
    pos_freqs: int = 2
    # Encoding for the (band, time) query coordinates; 2^3 cycles resolves a 16x6 grid.
    tf_freqs: int = 4
    mic_embed_dim: int = 8
    hidden: int = 128
    layers: int = 4
    skip_at: int = 2            # re-inject the input half-way (NeRF-style)
    # Constant axes (X and Z in the current grid) collapse to 0 instead of dividing by zero.
    min_pos_half_range_m: float = 1.0


@dataclass
class TrainConfig:
    seed: int = 0
    # Interpolation hold-out: every 5th Y position. Edge positions stay in training because
    # the model cannot be expected to extrapolate from a 1-D line of 23 points.
    val_positions: tuple = ((1.15, 0.4, 0.87), (1.15, 0.9, 0.87), (1.15, 1.4, 0.87), (1.15, 1.9, 0.87))
    steps: int = 4000
    batch_points: int = 16384   # (position, mic, freq, time) queries per step
    lr: float = 5e-4
    weight_decay: float = 1e-4
    eval_every: int = 250
    checkpoint_file: str = os.path.join(_HERE, "data", "ir_field.pt")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)
