"""Builds train_wake_word.ipynb from the cell sources below, and syntax-checks every code cell.
Run: python scripts/build_notebook.py   (from the repository root)"""
import ast
import json
import os

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(r'''
# Train a custom wake word (openWakeWord) on today's Google Colab

This notebook trains a small, fast wake-word model (an `.onnx` file of a few hundred KB) with
[openWakeWord](https://github.com/dscripka/openWakeWord). Its original training notebook no longer runs on
today's Colab (Python 3.13, new torchaudio, moved datasets); this one does. See the README for what changed and why.

**You bring the clips** (a zip with four folders, see the README): positive clips of your phrase and negative
clips of similar-sounding phrases. Piper (the original clip generator) is not used.

**Before you start**
1. Runtime -> Change runtime type -> **T4 GPU** -> Save.
2. Add a free Hugging Face **Read** token as a Colab secret named `HF_TOKEN` (key icon on the left) and switch
   on "Notebook access" -- without it Hugging Face rate-limits the downloads.
3. Keep this tab open. "Disconnect and delete runtime" deletes every downloaded file.

Run the cells from top to bottom, one at a time.
''')

code(r'''
## 0. Your settings -- the only cell you need to edit.
MODEL_NAME = "hey_nova"            # output file: my_custom_model/<MODEL_NAME>.onnx
TARGET_PHRASE = "hey nova"         # your phrase, spelled the way it should sound
CLIPS_ZIP = "clips.zip"            # your zip (positive_train, positive_test, negative_train, negative_test)
TRAINING_STEPS = 30000             # maximum training steps
AUGMENTATION_ROUNDS = 2            # how many times noise/room echo is mixed into each clip
''')

code(r'''
## 1. Packages for Python 3.13. The original notebook pins packages (e.g. tensorflow-cpu 2.8.1) that no
## longer install, so openWakeWord comes from PyPI without its pins, plus what augmentation and training need.
import os
if os.path.isdir("openwakeword"):        # a folder with this name would shadow the installed package
    os.rename("openwakeword", "openwakeword_old_clone")
!pip install -q --no-deps "openwakeword==0.6.0"
!pip install -q onnxruntime scikit-learn requests soundfile torchinfo torchmetrics pronouncing audiomentations torch-audiomentations acoustics mutagen speechbrain onnx onnxscript
import openwakeword, openwakeword.data, openwakeword.utils
print("ok -- packages ready")   # a warning about tflite-runtime can be ignored (only .onnx is used)
''')

code(r'''
## 2. Code and data (10-20 minutes; uses the HF_TOKEN secret).
## - openWakeWord's code (training script + default settings), cloned as "oww_repo" so it cannot shadow the package
## - openWakeWord's two feature models (audio -> features)
## - room impulse responses (MIT), background noise and music (MUSAN copy on Hugging Face)
## - precomputed negative features (ACAV100M, ~2,000 h) and a validation set (~11 h) for false-positive estimates
## - an empty stand-in for Piper: the training script imports it on every run, but no clips are generated here
import os, glob, subprocess
from huggingface_hub import snapshot_download, hf_hub_download

if not os.path.exists("oww_repo/openwakeword/train.py"):
    subprocess.run("git clone -q https://github.com/dscripka/openWakeWord.git oww_repo", shell=True, check=True)

openwakeword.utils.download_models(model_names=["__none__"])

snapshot_download("davidscripka/MIT_environmental_impulse_responses", repo_type="dataset",
                  allow_patterns=["16khz/**"], local_dir="mit_rirs_hf", max_workers=2)
os.makedirs("mit_rirs", exist_ok=True)
for f in glob.glob("mit_rirs_hf/16khz/**/*.wav", recursive=True):
    os.replace(f, os.path.join("mit_rirs", os.path.basename(f)))

snapshot_download("FluidInference/musan", repo_type="dataset",
                  allow_patterns=["noise/**", "music/**"], local_dir="musan_hf", max_workers=2)
for src, dst in (("musan_hf/noise", "bg_noise"), ("musan_hf/music", "bg_music")):
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(f"{src}/**/*.*", recursive=True):
        if f.lower().endswith((".wav", ".flac", ".mp3")):
            out = os.path.join(dst, os.path.splitext(os.path.basename(f))[0] + ".wav")
            if not os.path.exists(out):
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f, "-ar", "16000", "-ac", "1", out])

for name in ("openwakeword_features_ACAV100M_2000_hrs_16bit.npy", "validation_set_features.npy"):
    if not os.path.exists(name):
        hf_hub_download("davidscripka/openwakeword_features", name, repo_type="dataset", local_dir=".")

os.makedirs("piper_stub", exist_ok=True)
STUB = """def generate_samples(*args, **kwargs):
    raise RuntimeError("Piper is not used: all clips must already be in the clip folders")
"""
with open("piper_stub/generate_samples.py", "w") as f:
    f.write(STUB)

print("room responses:", len(os.listdir("mit_rirs")), "| noise:", len(os.listdir("bg_noise")),
      "| music:", len(os.listdir("bg_music")),
      "| features:", [os.path.exists(n) for n in ("openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
                                                   "validation_set_features.npy")])
''')

code(r'''
## 3. torchaudio fix. torchaudio 2.9+ removed torchaudio.info and moved torchaudio.load to another package;
## torch_audiomentations (noise mixing) and speechbrain (room echo) still call them. A small replacement based on
## soundfile is installed as a .pth file next to torchaudio, so it is active in every Python process -- including
## the training script, which runs as a separate process. All clips here are WAV, which soundfile reads.
## It also removes feature files left by an interrupted earlier run (otherwise Step A says "already exist" and skips).
import os, sys, glob, torchaudio

site_dir = os.path.dirname(os.path.dirname(torchaudio.__file__))
SHIM = """
try:
    import types, torch, soundfile as _sf, torchaudio as _ta
    def _info(path, *args, **kwargs):
        i = _sf.info(str(path))
        return types.SimpleNamespace(num_frames=i.frames, sample_rate=i.samplerate, num_channels=i.channels)
    def _load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, *args, **kwargs):
        data, sr = _sf.read(str(path), start=frame_offset, frames=num_frames, dtype="float32", always_2d=True)
        t = torch.from_numpy(data)
        return (t.T.contiguous() if channels_first else t), sr
    _ta.info = _info
    _ta.load = _load
except Exception as e:
    print("torchaudio shim not applied:", e)
"""
with open(os.path.join(site_dir, "wakeword_torchaudio_shim.py"), "w") as f:
    f.write(SHIM)
with open(os.path.join(site_dir, "wakeword_torchaudio_shim.pth"), "w") as f:
    f.write("import wakeword_torchaudio_shim\n")
!{sys.executable} -c "import torchaudio; print('shim ok:', torchaudio.info.__module__)"

for f in glob.glob(f"my_custom_model/{MODEL_NAME}/*.npy"):
    os.remove(f)
''')

code(r'''
## 4. Your clips. First drag your zip into the Files panel on the left and wait until the upload has finished
## (the spinner stops). The four folders go to my_custom_model/<MODEL_NAME>/.
import os, subprocess
print("zip size (bytes):", os.path.getsize(CLIPS_ZIP))
target = f"my_custom_model/{MODEL_NAME}"
subprocess.run(f"rm -rf {target} && mkdir -p {target} && unzip -q -o {CLIPS_ZIP} -d {target}", shell=True, check=True)
COUNTS = {d: len(os.listdir(f"{target}/{d}")) for d in ("positive_train", "positive_test", "negative_train", "negative_test")}
print(COUNTS)
assert all(COUNTS.values()), "every folder needs clips"
''')

code(r'''
## 5. Training settings -> my_model.yaml (Steps A and B read this file).
## The script generates clips only when a folder holds at most 95% of n_samples / n_samples_val; these are set
## just below your counts, so it never tries to generate (Piper is not used).
import yaml
config = yaml.load(open("oww_repo/examples/custom_model.yml").read(), yaml.Loader)

config["target_phrase"] = [TARGET_PHRASE]
config["model_name"] = MODEL_NAME
config["n_samples"] = max(1, int(min(COUNTS["positive_train"], COUNTS["negative_train"]) * 0.9))
config["n_samples_val"] = max(1, int(min(COUNTS["positive_test"], COUNTS["negative_test"]) * 0.9))
config["augmentation_rounds"] = AUGMENTATION_ROUNDS
config["steps"] = TRAINING_STEPS
config["piper_sample_generator_path"] = "./piper_stub"
config["rir_paths"] = ["./mit_rirs"]
config["background_paths"] = ["./bg_noise", "./bg_music"]
config["background_paths_duplication_rate"] = [1, 1]
config["false_positive_validation_data_path"] = "validation_set_features.npy"
config["feature_data_files"] = {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}
config["output_dir"] = "./my_custom_model"

with open("my_model.yaml", "w") as f:
    yaml.dump(config, f)
print("settings saved:", {k: config[k] for k in ("model_name", "n_samples", "n_samples_val", "steps")})
''')

code(r'''
## Step A: mix noise and room echo into the clips, then compute features (10-30 minutes).
## The many FutureWarning / CUDAExecutionProvider lines are warnings only.
import sys
!{sys.executable} oww_repo/openwakeword/train.py --training_config my_model.yaml --augment_clips
''')

code(r'''
## Step B: train the model (20-40 minutes). The last lines show accuracy, recall and false positives per hour.
## At the very end the script tries a .tflite conversion and fails with "No module named 'onnx_tf'" -- ignore it:
## the .onnx model is saved before that. Warnings about ONNX opset 18 are fine too (onnxruntime runs opset 18).
!{sys.executable} oww_repo/openwakeword/train.py --training_config my_model.yaml --train_model
''')

code(r'''
## 6. Download the model as ONE file. The new PyTorch exporter writes the weights to a separate
## "<name>.onnx.data" file; here they are merged into a single self-contained .onnx before download.
import os, onnx
from google.colab import files
src = f"my_custom_model/{MODEL_NAME}.onnx"
print(os.listdir("my_custom_model"))
model = onnx.load(src)                                   # also reads the .onnx.data next to it
single = src.replace(".onnx", "_single.onnx")
onnx.save_model(model, single, save_as_external_data=False)
print("single file:", single, os.path.getsize(single), "bytes")
files.download(single)
''')

if __name__ == "__main__":
    for i, c in enumerate(CELLS):
        if c["cell_type"] == "code":     # shell (!) lines are IPython syntax: comment them out for the check
            ast.parse("\n".join(("#" + l) if l.lstrip().startswith("!") else l for l in c["source"].split("\n")))
    nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "train_wake_word.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"train_wake_word.ipynb written ({len(CELLS)} cells, all code cells parse)")
