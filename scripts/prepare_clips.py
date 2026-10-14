"""Turn raw clips into the zip the notebook expects.

1. Converts every clip to 16 kHz mono WAV (needs ffmpeg on PATH).
2. Drops broken clips: shorter than 0.5 s, longer than 3.5 s, or nearly silent.
3. Splits positives and negatives into train / test and writes one zip with four folders:
   positive_train, positive_test, negative_train, negative_test.

Example:
  python scripts/prepare_clips.py --positive data/clips/positive --negative data/clips/negative --out data/clips.zip
"""
import argparse
import os
import random
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.io import wavfile

AUDIO = (".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".webm")


def convert_and_check(src, dst):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ar", "16000", "-ac", "1", dst], check=True)
    sr, x = wavfile.read(dst)
    x = x.astype(np.float32) / 32768
    duration = len(x) / sr
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    return 0.5 <= duration <= 3.5 and rms >= 0.01


def collect(dirs, tmp, tag):
    files = [os.path.join(d, f) for d in dirs for f in sorted(os.listdir(d)) if f.lower().endswith(AUDIO)]
    outs = [os.path.join(tmp, f"{tag}_{i:05d}.wav") for i in range(len(files))]
    with ThreadPoolExecutor(8) as ex:
        good = list(ex.map(convert_and_check, files, outs))
    kept = [o for o, g in zip(outs, good) if g]
    print(f"{tag}: {len(files)} clips, kept {len(kept)}, dropped {len(files) - len(kept)}")
    return kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--positive", nargs="+", required=True, help="folders with clips of your phrase")
    p.add_argument("--negative", nargs="+", required=True, help="folders with clips of similar phrases")
    p.add_argument("--out", default="clips.zip")
    p.add_argument("--test-share", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=9)
    a = p.parse_args()
    random.seed(a.seed)
    with tempfile.TemporaryDirectory() as tmp:
        groups = {}
        for tag, dirs in (("positive", a.positive), ("negative", a.negative)):
            kept = collect(dirs, tmp, tag)
            random.shuffle(kept)
            n_test = max(1, int(len(kept) * a.test_share))
            groups[f"{tag}_test"], groups[f"{tag}_train"] = kept[:n_test], kept[n_test:]
        with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
            for name, files in groups.items():
                for f in files:
                    z.write(f, f"{name}/{os.path.basename(f)}")
    print({k: len(v) for k, v in groups.items()}, f"-> {a.out} ({os.path.getsize(a.out)} bytes)")


if __name__ == "__main__":
    main()
