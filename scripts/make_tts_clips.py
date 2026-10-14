"""Make training clips with Gemini text-to-speech: many voices, accents, speeds and styles.

Use it for positive clips (your phrase) and for negative clips (similar-sounding phrases the model must ignore).
Needs a Gemini API key in the environment variable GEMINI_API_KEY (it is never printed).

Examples:
  python scripts/make_tts_clips.py voices --lang en-IN                      # list the voices for one language
  python scripts/make_tts_clips.py make --n 1000 --out data/clips/positive --phrases "Hey Nova" "Hey, Nova" "Hey Nova!"
  python scripts/make_tts_clips.py make --n 1000 --out data/clips/negative --phrases "Hey Noah" "Hey Siri" "Nova" "Hey"
  python scripts/make_tts_clips.py make --n 500 --out data/clips/positive_in --langs en-IN hi-IN --phrases "Hey Nova"

Tips:
- Spell the phrase the way it should sound and check a few clips by ear (a TTS voice may say a new word
  differently than you do).
- Clips come as 24 kHz WAV; run prepare_clips.py next to convert them to 16 kHz and drop broken ones.
"""
import argparse
import base64
import os
import random
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://generativelanguage.googleapis.com/v1beta"
MODEL = os.environ.get("TTS_MODEL", "gemini-3.8-flash-lite-tts")

PREBUILT = ["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
            "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
            "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird",
            "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"]
STYLES = ["neutral", "speaking rapidly", "speaking slowly", "casual and relaxed", "tired", "cheerful",
          "quiet, almost whispering", "loud, calling from across the room", "questioning", "firm"]


def headers():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set the GEMINI_API_KEY environment variable first.")
    return {"x-goog-api-key": key, "Content-Type": "application/json"}


def synth(text, voice, style, path):
    """One clip -> path. Returns "ok ..." or an error text; network errors are returned, not raised."""
    body = {"model": MODEL,
            "input": [{"type": "user_input", "content": [{"type": "text", "text": text,
                       "annotations": [{"type": "speech_metadata", "style": style}]}]}],
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": voice}]}}
    try:
        r = requests.post(f"{API}/interactions", headers=headers(), json=body, timeout=30)
    except requests.RequestException as e:
        return f"retry: network {type(e).__name__}"
    if r.status_code == 429:
        return "retry: rate limited"
    if r.status_code != 200:
        return f"HTTP {r.status_code}: {r.text[:200]}"
    audio = [c for s in r.json().get("steps", []) if s.get("type") == "model_output"
             for c in s.get("content", []) if c.get("type") == "audio"]
    if not audio:
        return "no audio in the response"
    with open(path, "wb") as f:
        f.write(base64.b64decode(audio[-1]["data"]))
    with wave.open(path) as w:
        return f"ok {w.getnframes() / w.getframerate():.2f} s"


def library_voices(lang):
    """Voice ids from the voice library for one language code (e.g. en-IN, hi-IN, en-GB)."""
    out, token = [], None
    while True:
        params = {"languageCode": lang, "pageSize": 1000}
        if token:
            params["pageToken"] = token
        d = requests.get(f"{API}/voices", headers=headers(), params=params, timeout=30).json()
        out += d.get("voices", [])
        token = d.get("nextPageToken")
        if not token:
            return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("voices")
    v.add_argument("--lang", required=True)
    m = sub.add_parser("make")
    m.add_argument("--n", type=int, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--phrases", nargs="+", required=True)
    m.add_argument("--langs", nargs="*", default=[], help="add library voices of these languages (e.g. en-IN hi-IN)")
    m.add_argument("--no-prebuilt", action="store_true", help="use only the library voices")
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--workers", type=int, default=6)
    a = p.parse_args()

    if a.cmd == "voices":
        for x in library_voices(a.lang):
            print(x["id"], x.get("gender"), x.get("accent"), "-", x.get("description", "")[:80])
        return

    pool = [] if a.no_prebuilt else list(PREBUILT)
    for lang in a.langs:
        pool += [x["id"] for x in library_voices(lang)]
    if not pool:
        sys.exit("No voices.")
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    jobs = [(i, random.choice(pool), random.choice(STYLES), random.choice(a.phrases)) for i in range(a.n)]

    def one(job):
        i, voice, style, phrase = job
        path = os.path.join(a.out, f"{i:05d}_{voice}.wav")
        if os.path.exists(path):        # a rerun continues where it stopped
            return "ok"
        for attempt in range(3):
            res = synth(phrase, voice, style, path)
            if not res.startswith("retry"):
                return res
            time.sleep(5 * (attempt + 1))
        return res

    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(one, jobs))
    bad = [r for r in results if not r.startswith("ok")]
    print(f"{a.n - len(bad)}/{a.n} clips in {a.out}; errors: {sorted(set(b[:80] for b in bad))[:5]}")


if __name__ == "__main__":
    main()
