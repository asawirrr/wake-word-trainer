# wake-word-trainer

Train your own wake word ("Hey Nova", "Hey Jarvis", ...) as a small, fast model that runs on a laptop or in the
browser. Built on [openWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka (Apache-2.0).

**Why this exists:** openWakeWord's own training notebook no longer runs on today's Google Colab (Python 3.13,
torchaudio 2.9+, moved or removed datasets). This repository has a notebook that does, plus scripts to make
training clips with a text-to-speech API instead of Piper, and to test the model on real voices.

> Status: the notebook has run end to end twice on Colab (T4 GPU, Python 3.13, October 2026). A model trained
> from about 2,500 synthetic positive clips detected 92% of real spoken phrases from several people, with no false
> wake in 10 minutes of ordinary office conversation. Your numbers depend on your phrase and clips -- always test
> with real voices (step 3). Lesson learned: piling on synthetic "almost the phrase" negatives cut look-alike wakes
> but also cost real-voice detection; real human recordings help more than more synthetic clips.

## What you get

| File | What it does |
|---|---|
| `train_wake_word.ipynb` | The Colab notebook: downloads the data, applies the fixes below, trains, gives you `<name>.onnx` |
| `scripts/make_tts_clips.py` | Makes clips with Gemini text-to-speech: dozens of voices and accents, speeds and styles |
| `scripts/prepare_clips.py` | Converts clips to 16 kHz, drops broken ones, writes the zip the notebook expects |
| `scripts/split_recording.py` | Cuts your own recording ("Hey Nova" x 20) into one clip per phrase, for testing |
| `scripts/test_model.py` | Measures the model: detection on real voices, false wakes per hour, look-alike phrases |
| `scripts/build_notebook.py` | Rebuilds the notebook from readable source (edit cells there) |

## What you need

**Accounts and tools**
- A Google account for [Colab](https://colab.research.google.com) (the free T4 GPU is enough; a paid plan keeps
  running when the tab is closed).
- A free [Hugging Face](https://huggingface.co) account and a **Read** token (for the data downloads).
- A Gemini API key for the synthetic clips (or another text-to-speech service -- adapt `make_tts_clips.py`).
- On your computer: Python 3.10+, `pip install -r requirements.txt`, and [ffmpeg](https://ffmpeg.org) on PATH.

**Clips for training (made with text-to-speech)**
- **Positive:** 2,000+ clips of your phrase, in many voices, accents, speeds and styles. Spell it the way your
  users say it, and include the other common pronunciations.
- **Negative:** 1,000+ clips of phrases that sound similar but must not wake it: other names with the same
  first word ("Hey Noah"), other assistants ("Hey Siri"), the name alone, the first word alone.

**Recordings of real people (for testing -- do not skip)**
- **Your voice:** say the phrase about **20 times** in one recording, with **random pauses of 2-5 seconds**
  (never under 2 s, so the clips can be cut apart). Vary it: normal, fast, slow, from a distance, tired, with a fan on.
- **2-3 other people:** about **10 times each**, in their own natural speed and tone -- do not coach them.
  **One file per person**, so you can see who it works for. (One file for everyone also works: 5-10 s of silence
  between people.)
- Background talk is fine, even useful, as long as the phrase is clearly louder.
- **Optional but valuable:** 15-20 minutes of your normal surroundings (office, meeting, TV) **without** the phrase,
  to count false wakes where it will really be used. Include the languages people speak around it.
- Any format works (phone voice notes, WhatsApp audio, WAV); ffmpeg converts them.
- Recordings of people are personal data: keep them in `data/` (git-ignored), never in the repository.

## Folder layout

```
train_wake_word.ipynb     the Colab notebook
scripts/                  clip making, preparation, testing, notebook builder
data/                     git-ignored: everything with audio
  recordings/             your raw recordings
  clips/positive/         positive clips (16 kHz)
  clips/negative/         negative clips (16 kHz)
  clips.zip               what you upload to the notebook
  test/real_positive/     real phrases cut from your recordings (split_recording.py)
  test/background/        long audio without the phrase (false-wake test)
  test/lookalike/         similar phrases (look-alike test)
models/                   git-ignored: trained .onnx models
local/                    git-ignored: your own notes
```

## How it works (short)

1. **Clips.** Positive clips say your phrase; negative clips say similar phrases ("Hey Noah", "Hey Siri", "Nova").
2. **Mixing.** The training script mixes background noise, music and room echo into every clip.
3. **Features.** Audio becomes a spectrogram, then a pre-trained speech-embedding model (Google, Apache-2.0)
   turns every 80 ms into 96 numbers. This part is not trained -- it is reused (transfer learning).
4. **Training.** A tiny neural network learns "phrase or not" on those numbers, against your negatives and
   ~2,000 hours of everyday audio, and checks itself on ~11 hours of other audio to estimate false wakes per hour.
5. **Result.** `<name>.onnx`, a few hundred KB. At run time: microphone -> features -> your model -> a score from
   0 to 1; above the threshold (0.5) it wakes.

## Step by step

### 1. Make the clips (on your computer)

```bash
pip install -r requirements.txt          # also needs ffmpeg on PATH
export GEMINI_API_KEY=...                 # Windows PowerShell: $env:GEMINI_API_KEY="..."

# positives: your phrase, spelled the way it should sound, a few ways of writing it
python scripts/make_tts_clips.py make --n 2000 --out data/clips/positive --langs en-IN en-GB \
    --phrases "Hey Nova" "Hey, Nova" "Hey Nova!" "hey nova?"

# negatives: phrases that sound similar but must NOT wake it, plus the name alone and the first word alone
python scripts/make_tts_clips.py make --n 1200 --out data/clips/negative --langs en-IN en-GB --seed 31 \
    --phrases "Hey Noah" "Hey Siri" "Hey Nora" "Hey no" "Nova" "Hey" "hey there" "okay nova" "Hey Google"

python scripts/prepare_clips.py --positive data/clips/positive --negative data/clips/negative --out data/clips.zip
```

Listen to a few clips first: a voice may pronounce a new word differently than you do. If so, change the spelling
("Hey Kay-la" instead of "Hey Kayla"). Cost: a few thousand short clips cost about a dollar or two
(check current Gemini prices).

### 2. Run the notebook on Colab

1. Open [colab.research.google.com](https://colab.research.google.com) -> File -> Upload notebook ->
   `train_wake_word.ipynb`.
2. Runtime -> Change runtime type -> **T4 GPU**.
3. Create a free Hugging Face token (Settings -> Access Tokens -> **Read**), add it as the Colab secret `HF_TOKEN`
   (key icon) and switch on "Notebook access". Delete the token when you are done.
4. Edit cell 0 (name, phrase, zip name). Run cells 1-3.
5. Drag `clips.zip` into the Files panel, wait until the upload has finished, run cells 4-5.
6. Run Step A (mixing + features, 10-30 min), then Step B (training, 20-40 min), then cell 6 to download the model
   as one self-contained file (`<name>_single.onnx`; the plain `<name>.onnx` is only a shell without its weights).

Keep the tab open, and never use "Disconnect and delete runtime" in the middle: a new runtime starts empty.
If that happens, run all cells again from the top.

### 3. Test it with real voices (on your computer)

```bash
python scripts/split_recording.py data/recordings/me.m4a --count 20 --out data/test/real_positive
python scripts/test_model.py models/hey_nova.onnx --positive data/test/real_positive \
    --background data/test/background --lookalike data/test/lookalike
```

Rough targets: detection of 90% or more of real phrases; one false wake per hour or fewer.

## What was fixed compared with the original notebook

| Problem on today's Colab | Fix in this notebook |
|---|---|
| Pinned packages (tensorflow-cpu 2.8.1, old datasets) do not install on Python 3.13 | openWakeWord from PyPI with `--no-deps`; only the packages the training needs, current versions |
| `pyarrow` has no `PyExtensionType` (old `datasets`) | `datasets` is no longer needed |
| AudioSet file returns 404; the FMA dataset no longer streams | Background noise and music from a MUSAN copy on Hugging Face |
| Piper's `piper_phonemize` does not install on Python 3.13, and `train.py` imports Piper on every run | All clips are given (positive and negative); a stand-in module replaces Piper |
| `torchaudio.info` removed and `torchaudio.load` moved (torchaudio 2.9+) | A small replacement based on `soundfile`, installed as a `.pth` file so the training process gets it too |
| A clone named `openwakeword` in the working folder shadows the installed package | Cloned as `oww_repo` |
| Hugging Face rate-limits anonymous downloads | Use a free Read token (`HF_TOKEN` secret) |
| An interrupted Step A leaves half-written feature files and the next run skips | Cell 3 removes old feature files |

## Licences -- read before commercial use

- **Code** in this repository: Apache-2.0 (see `LICENSE`). Based on openWakeWord, Apache-2.0 -- see `NOTICE`.
- **openWakeWord's feature models** (melspectrogram, speech embedding): Apache-2.0.
- **Training data the notebook downloads -- check each before a commercial product:**
  - ACAV100M precomputed features (negative training data): derived from YouTube videos; no clear licence.
    openWakeWord's own pre-trained models are non-commercial (CC BY-NC-SA) for reasons like this.
  - MUSAN (noise, music): Creative Commons / public domain per its authors; the Hugging Face copy is a community upload.
  - MIT room impulse responses: licence listed as unknown on Hugging Face.
- **Your TTS clips:** subject to the text-to-speech provider's terms.

For a product, replace the ACAV100M features with negatives you have the rights to (for example LibriSpeech,
CC BY 4.0, or Mozilla Common Voice, CC0) and keep the credits in `NOTICE`.
