import argparse
import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

# Plain ASCII output, flushed on every line: Git Bash on Windows pipes stdout
# (block-buffered, cp1252), which would otherwise delay the cue or crash on
# non-ASCII characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace", line_buffering=True)


def say(msg):
    print(msg, flush=True)


def main():
    # 1. Parse arguments from capture.sh
    parser = argparse.ArgumentParser(description="Hardware capture for Phased Array.")
    parser.add_argument("--code", type=str, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--countdown", type=int, default=3,
                        help="seconds of countdown before recording starts")
    args = parser.parse_args()

    # Configuration: Ensure this matches your hardware!
    SAMPLE_RATE = 44100
    CHANNELS = 4  # Mic1 = playback loopback, Mic2-Mic4 = array

    # 2. Setup Directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "MICRECORD", args.code, "INDIV")
    os.makedirs(output_dir, exist_ok=True)

    say(f"Session {args.code}: get ready, wait for the RECORDING cue before pressing play.")

    try:
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
            say(f"  -> Saved {filename}  (peak {np.max(np.abs(single_channel_data)):.2f})")

        say("Hardware capture complete.")

    except Exception as e:
        say(f"ERROR: Failed to record audio. Details:\n{e}")
        say("\nTROUBLESHOOTING: Run this in a python shell to find your mic's device ID:")
        say("   import sounddevice as sd; print(sd.query_devices())")
        say("   Then add `device=YOUR_DEVICE_ID` to the sd.rec() function.")
        sys.exit(1)

if __name__ == "__main__":
    main()
