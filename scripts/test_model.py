"""Measure a trained model on your own audio.

- Detection: folders of clips that contain the phrase (ideally real voices) -> how many are detected.
- False wakes: long recordings WITHOUT the phrase (talk, meetings, TV) -> wakes per hour.
- Look-alikes: clips of similar phrases -> how many wrongly wake it.

Example:
  python scripts/test_model.py hey_nova.onnx --positive test/real_positive --background test/office.m4a --lookalike test/similar
Needs: pip install openwakeword onnxruntime numpy scipy  (and ffmpeg on PATH for non-WAV files).
"""
import argparse
import glob
import os
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile

AUDIO = (".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".webm")


def load16k(path, tmp):
    out = os.path.join(tmp, os.path.basename(path) + ".wav")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", path, "-ar", "16000", "-ac", "1", out], check=True)
    return wavfile.read(out)[1].astype(np.int16)


def scores(model, audio):
    """The model's score for every 80 ms frame of the audio (1 s of silence padded before and after)."""
    model.reset()
    pad = np.zeros(16000, dtype=np.int16)
    audio = np.concatenate([pad, audio, pad])
    out = []
    for i in range(0, len(audio) - 1280 + 1, 1280):
        out.append(max(model.predict(audio[i:i + 1280]).values()))
    return np.array(out)


def files(paths):
    out = []
    for p in paths:
        out += [f for f in glob.glob(os.path.join(p, "**", "*"), recursive=True) if f.lower().endswith(AUDIO)] \
            if os.path.isdir(p) else [p]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--positive", nargs="*", default=[])
    p.add_argument("--background", nargs="*", default=[])
    p.add_argument("--lookalike", nargs="*", default=[])
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()

    from openwakeword.model import Model
    from openwakeword import utils
    utils.download_models(model_names=["__none__"])     # the two feature models
    model = Model(wakeword_models=[a.model], inference_framework="onnx")

    with tempfile.TemporaryDirectory() as tmp:
        if a.positive:
            fs = files(a.positive)
            peaks = [scores(model, load16k(f, tmp)).max() for f in fs]
            hit = sum(s >= a.threshold for s in peaks)
            print(f"Detection: {hit}/{len(fs)} ({100 * hit / max(1, len(fs)):.0f}%) at threshold {a.threshold}; "
                  f"median peak score {np.median(peaks):.2f}")
            for f, s in zip(fs, peaks):
                if s < a.threshold:
                    print(f"  missed: {os.path.basename(f)} (peak {s:.2f})")
        if a.lookalike:
            fs = files(a.lookalike)
            peaks = [scores(model, load16k(f, tmp)).max() for f in fs]
            bad = [(f, s) for f, s in zip(fs, peaks) if s >= a.threshold]
            print(f"Look-alikes that woke it: {len(bad)}/{len(fs)}")
            for f, s in bad[:20]:
                print(f"  {os.path.basename(f)} (peak {s:.2f})")
        if a.background:
            total_s, wakes = 0.0, 0
            for f in files(a.background):
                audio = load16k(f, tmp)
                total_s += len(audio) / 16000
                s = scores(model, audio)
                above, i = s >= a.threshold, 0
                while i < len(above):            # one wake per crossing, then 2 s of quiet before the next
                    if above[i]:
                        wakes += 1
                        i += 25
                    else:
                        i += 1
            hours = total_s / 3600
            print(f"False wakes: {wakes} in {hours:.2f} h = {wakes / max(hours, 1e-9):.2f} per hour")


if __name__ == "__main__":
    main()
