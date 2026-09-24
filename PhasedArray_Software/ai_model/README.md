# Array-response model (phantom array)

Predicts what the microphone array (Mic2–Mic4) would record if it were placed at a
coordinate `(x, y, z)` in the room, for any source audio (e.g. a lecturer's voice).

```
position (x,y,z) ──► IRFieldNet ──► per-mic band-energy envelope ──► synthesized impulse response ──► ⊛ dry audio ──► predicted mic signals
```

## Quick start (from `PhasedArray_Software/`)

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # torch w/ CUDA: see pytorch.org
.venv/Scripts/python -m ai_model.prepare_dataset          # validate MICRECORD/Grid_* and build data/ir_dataset.pt
.venv/Scripts/python -m ai_model.train                    # ~20 s on an RTX 4060
.venv/Scripts/python -m ai_model.render --pos 1.15 1.25 0.87 --source ../MICRECORD/dry_sweep.wav --out pred/
```

`prepare_dataset --check-only` only validates and writes `data/manifest.csv` and
`data/validation_report.txt`. Run it after every capture session.

## Why the model predicts impulse responses (not raw recordings)

Each grid take is a 3 s exponential sine sweep (20 Hz–20 kHz, `dry_sweep.wav`) recorded
by 3 mics. Deconvolving by the dry sweep gives each mic's **room impulse response (IR)**,
and `predicted IR ⊛ any audio` = what that mic would hear. This matters because:

* The sweep playback was **not synchronized** with recording. It started anywhere from 713 ms
  before to 535 ms after the recording began, so the raw 5 s waveforms are misaligned
  from take to take. A model trained on raw audio would have to learn random offsets. IR
  extraction removes the offset (only timing *between* mics in the same take is kept).
* One IR per mic per position is a compact, source-independent target (0.35 s at 16 kHz).
* The old `simulate_mic.py` design mapped spectrogram→spectrogram and needed paired
  recordings of the same content, which doesn't exist in this dataset.

## Chosen parameters (all in `config.py`) and the evidence for each

| Parameter | Value | Why |
|---|---|---|
| Model sample rate | 16 kHz | Speech application; recordings contain no sweep energy above ~9–10 kHz |
| Band | 500–7500 Hz | Measured IR dynamic range per octave: 125 Hz **1 dB**, 250 Hz **4 dB**, 500 Hz 17 dB, 1 kHz 28 dB, 2–8 kHz 42–54 dB. Below 500 Hz is hum/noise |
| IR length | 0.35 s (+5 ms pre-roll) | IRs decay ~20 dB per 125 ms (RT60 ≈ 0.38 s) and hit the ~55 dB noise floor by ~0.35 s |
| Target | log energy, 16 mel bands × 64 ms frames (6 frames) | See "target resolution" below |
| Network | 4-layer MLP, 128 wide, skip at layer 2, **60k params** | 64×3 / 128×4 / 256×6 → 1.05 / 1.02 / 1.01 dB; bigger is not better |
| Position encoding | Fourier features, 2 frequencies | 0/1/2/3/4 freqs → 1.030 / 1.011 / 1.013 / 1.040 / 1.11 dB (5 seeds). Data only supports smooth spatial variation |
| Band/time encoding | 4 frequencies | Enough to address a 16×6 grid |
| Mic embedding | 8-dim learned, one per mic | Shares one network across mics |
| Loss | L1 on normalized log band energy | Robust to the few noisy cells |
| Optimizer | AdamW, lr 5e-4, one-cycle, 4000 steps, 16k queries/step | Converges by ~1500 steps; weight decay had no effect |
| Validation | hold out Y = 0.4, 0.9, 1.4, 1.9 | Tests interpolation between grid points; ends are kept (no extrapolation data) |

### Target resolution

The first design predicted raw STFT bins (31 Hz × 8 ms). The network could not beat simply
predicting the **average of all positions** (5.19 vs 5.28 dB), and nearest-neighbour was
*worse* than the average. The reason is physics: in a reverberant room, fine spectral detail
of an IR changes randomly within 0.1 m (diffuse-field fluctuations, ~5.6 dB spread). Only
coarser quantities are smooth in space:

| Feature | Neighbour correlation across Y (1 = smooth) |
|---|---|
| raw band × 8 ms cells | 0.10 – 0.16 (random) |
| Mic3 total level | 0.70 |
| early/late energy ratio (C50) | ~0.7 (Mic2, Mic3) |
| direct-sound level | 0.4 – 0.6 |

Sweeping the target resolution, linear interpolation of neighbours only starts to beat the
global mean at **8–16 bands × 64 ms**, so that's the resolution the data supports. Rendering
resynthesizes the random fine structure as shaped noise, which is standard for late reverb
and inaudible as an error.

## Current results

| Predictor (held-out positions) | Band-envelope error |
|---|---|
| Average of training positions | 1.11 dB |
| Nearest training position | 1.28 dB |
| Linear interpolation | 1.03 dB |
| **IRFieldNet** | **1.00 dB** |

The model does learn the one strong spatial effect in the data. Mic3's direct sound rises
+4–8 dB relative to Mic2 for Y ≈ 1.4–1.9 m, and the model reproduces it, including at held-out
points, though smoothed. **The honest reading is that with a 1-D line of 23 points the network
is only as good as linear interpolation.** Its value will come from 2-D/3-D, irregular grids,
where simple interpolation stops working. Data is now the bottleneck, not the model.

## What the model cannot do yet

* **X and Z are constant** in the dataset (X = 1.15, Z = 0.87), so any other X/Z is
  extrapolation (`render.py` warns).
* **Exact inter-mic timing/phase is not modelled.** Rendered IRs have the right level,
  colouration and decay, but not the precise delays that delay-and-sum beamforming relies on.
  Spatial Nyquist for a 0.1 m grid is ~1.7 kHz, and in these recordings the direct sound is
  ~15 dB below the reverb at Mic2/Mic4, so delays can't be measured reliably from this data.
  A geometry-based delay term (needs mic/speaker coordinates) is the planned next head.
* Below 500 Hz nothing is predicted (no usable measurement there).

## Data-collection recommendations (in priority order)

1. **Play the sweep from the capture script** (`sounddevice.playrec`), and ideally record a
   loopback of the playback signal on a spare channel (Mic1's input is free). This fixes sync
   and makes absolute arrival times, and therefore beamforming delays, learnable.
2. **Cover 2-D (X and Y), then Z**, over the area the lecturer actually moves in. 0.2–0.3 m
   spacing is enough for band-energy targets (e.g. a 3 m × 2 m area at 0.25 m ≈ 100 points).
3. **Repeat 2–3 takes at ~5 positions.** Right now measurement noise can't be separated from real
   spatial variation (e.g. Y = 1.8 breaks the Mic3 trend of its neighbours). Repeats tell us
   the best error any model could reach.
4. **Improve low-frequency SNR:** a louder or longer sweep (10 s ESS ≈ +5 dB), several averaged
   sweeps, and find the ~120 Hz hum (likely a ground loop; it's strongest on Mic4).
5. **Match channel gains:** Mic4 is ~9 dB hotter than Mic2. The model learns gains as-is, but
   calibrated gains make results physically interpretable.
6. **Log geometry** in a sidecar file per session: speaker position, and each mic's offset and
   orientation relative to the array reference point.
7. **Check Mic3:** the array moved as a rigid unit, yet only Mic3's level/direct sound changes
   with Y. Check its orientation and polar pattern relative to the others.

## Files

| File | Purpose |
|---|---|
| `config.py` | Every parameter, with the measurement that justified it |
| `acoustics.py` | Sweep deconvolution, onset detection, IR cropping, quality metrics |
| `prepare_dataset.py` | Validates the grid recordings, writes manifest/report, builds the IR cache |
| `dataset.py` | Band-energy targets, position normalization, train/val split |
| `model.py` | `IRFieldNet` |
| `train.py` | Training + baselines; saves `data/ir_field.pt` |
| `render.py` | Position + dry audio → predicted per-mic audio and IRs |
