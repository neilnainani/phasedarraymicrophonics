import argparse
import os
import sys
import time
import numpy as np

# The Behringer UMC driver's ASIO device gives all 4 inputs at the requested
# rate with no Windows mixing/resampling; sounddevice only lists ASIO devices
# when this is set before it is imported.
os.environ.setdefault("SD_ENABLE_ASIO", "1")
import sounddevice as sd
import soundfile as sf

# Plain ASCII output, flushed on every line: Git Bash on Windows pipes stdout
# (block-buffered, cp1252), which would otherwise delay the cue or crash on
# non-ASCII characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace", line_buffering=True)


# Recordings go to MICRECORD/<RECORDINGS_FOLDER>/<code>/INDIV. Keep in sync with
# RECORDINGS_FOLDER in capture.sh and recordings_dir in ai_model/config.py.
RECORDINGS_FOLDER = "Oct2026recordings"


# Interface used when --device is not given: the first input whose name contains
# one of these, in order (the ASIO device first, then the 4-input Windows device).
PREFERRED_DEVICES = ("UMC ASIO", "IN 1-4 (BEHRINGER UMC")


def say(msg):
    print(msg, flush=True)


def pick_input_device(requested, channels):
    """Return the input device index to record from."""
    devices = sd.query_devices()
    if requested is not None:
        if requested.isdigit():
            return int(requested)
        names = [requested]
    else:
        names = PREFERRED_DEVICES
    for name in names:
        for i, dev in enumerate(devices):
            if name.lower() in dev["name"].lower() and dev["max_input_channels"] >= channels:
                return i
    if requested is not None:
        raise RuntimeError(f"no input device matching '{requested}' with {channels} channels")
    return sd.default.device[0]   # fall back to the Windows default input


def main():
    # 1. Parse arguments from capture.sh
    parser = argparse.ArgumentParser(description="Hardware capture for Phased Array.")
    parser.add_argument("--code", type=str, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--countdown", type=int, default=3,
                        help="seconds of countdown before recording starts")
    parser.add_argument("--folder", type=str, default=RECORDINGS_FOLDER,
                        help="session folder inside MICRECORD/ to save into")
    parser.add_argument("--device", type=str, default=None,
                        help="input device index or name (default: the UMC404HD, ASIO first)")
    args = parser.parse_args()

    # Configuration: Ensure this matches your hardware!
    SAMPLE_RATE = 44100
    CHANNELS = 4  # Mic1 = playback loopback, Mic2-Mic4 = array

    # 2. Setup Directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "MICRECORD", args.folder, args.code, "INDIV")
    os.makedirs(output_dir, exist_ok=True)

    say(f"Session {args.code}: get ready, wait for the RECORDING cue before pressing play.")

    try:
        device = pick_input_device(args.device, CHANNELS)
        dev = sd.query_devices(device)
        say(f"Input device {device}: {dev['name']} ({sd.query_hostapis(dev['hostapi'])['name']})")

        # 3. Countdown here rather than in capture.sh, so it runs after the
        # (slow, 1-2 s on Windows) library imports and the cue below is
        # printed only once the input stream is actually running.
        for i in range(args.countdown, 0, -1):
            say(f"  {i}...")
            time.sleep(1)

        # sd.rec starts the stream and returns immediately; sd.wait blocks until done
        recording = sd.rec(int(args.duration * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE,
                           channels=CHANNELS,
                           device=device,
                           dtype='float32')
        say(f">>> RECORDING for {args.duration:g} seconds! Press play now. <<<")
        sd.wait()

        # 4. Split and save the channels
        for i in range(CHANNELS):
            mic_num = i + 1
            filename = f"Mic{mic_num}_{args.code}.wav"
            filepath = os.path.join(output_dir, filename)

            # Extract the specific channel (column) from the numpy array
            single_channel_data = recording[:, i]

            # Save to standard 16-bit PCM WAV
            sf.write(filepath, single_channel_data, SAMPLE_RATE, subtype='PCM_16')
            peak = np.max(np.abs(single_channel_data))
            say(f"  -> Saved {filename}  (peak {peak:.2f})")
            if peak < 1e-3:
                say(f"     WARNING: Mic{mic_num} is silent - check its cable, gain knob and 48 V")

        say("Hardware capture complete.")

    except Exception as e:
        say(f"ERROR: Failed to record audio. Details:\n{e}")
        say("\nTROUBLESHOOTING: list the devices with  python -m sounddevice")
        say("   then pass the right one with  --device <index or name>.")
        say("   The ASIO device fails if another program (DAW, browser tab) is using it.")
        sys.exit(1)

if __name__ == "__main__":
    main()
