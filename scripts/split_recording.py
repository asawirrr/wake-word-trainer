"""Cut a recording of a person saying the phrase many times into one clip per phrase.

Record yourself (phone or laptop, any format) saying the phrase N times with 2-5 s pauses. The script finds the
loudness threshold that gives exactly N pieces, so background talk or noise is handled. Needs ffmpeg on PATH.
Use these clips to TEST the model (real voices), not only synthetic ones.

Example:
  python scripts/split_recording.py my_recording.m4a --count 20 --out test/real_positive
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile


def segments(rms, threshold, gap_frames):
    on = rms > threshold
    found, start, quiet = [], None, 0
    for i, v in enumerate(on):
        if v:
            start = i if start is None else start
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= gap_frames:
                found.append((start, i - quiet))
                start, quiet = None, 0
    if start is not None:
        found.append((start, len(on) - 1))
    return [(a, b) for a, b in found if 0.3 <= (b - a + 1) * 0.02 <= 3.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("recording")
    p.add_argument("--count", type=int, required=True, help="how many times the phrase was said")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "full.wav")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.recording, "-ar", "16000", "-ac", "1", wav], check=True)
        sr, x = wavfile.read(wav)
    x = x.astype(np.float32) / 32768
    hop = int(0.02 * sr)
    rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2)) for i in range(len(x) // hop)])
    for gap in (40, 30, 25, 20):                     # 0.8 .. 0.4 s of quiet between phrases
        for q in range(95, 20, -1):
            segs = segments(rms, np.percentile(rms, q), gap)
            if len(segs) == a.count:
                base = os.path.splitext(os.path.basename(a.recording))[0]
                for k, (s, e) in enumerate(segs, 1):
                    lo, hi = max(0, (s - 15) * hop), min(len(x), (e + 15) * hop)
                    wavfile.write(os.path.join(a.out, f"{base}_{k:02d}.wav"), sr, (x[lo:hi] * 32767).astype(np.int16))
                print(f"{a.count} clips -> {a.out}; starts (s):", [round(s * 0.02, 1) for s, _ in segs])
                return
    print(f"Could not find exactly {a.count} phrases; check the count, or record with longer pauses.")


if __name__ == "__main__":
    main()
