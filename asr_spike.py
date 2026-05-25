import os, base64
from pathlib import Path

import google.generativeai as genai

ENV_FILE = Path(__file__).resolve().parent / "env"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load_env_file(ENV_FILE)

api_key = os.environ.get("NUXT_GEMINI_API_KEY")
model_name = os.environ.get("NUXT_GEMINI_MODEL")
if not api_key or not model_name:
    raise RuntimeError(
        "Missing NUXT_GEMINI_API_KEY or NUXT_GEMINI_MODEL in env file or environment"
    )

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name)

import mimetypes

MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".m4a": "audio/mp4",
}

def guess_mime(audio_path: str) -> str:
    ext = Path(audio_path).suffix.lower()
    if ext in MIME_BY_EXT:
        return MIME_BY_EXT[ext]
    guessed, _ = mimetypes.guess_type(audio_path)
    return guessed or "application/octet-stream"
    
def asr_with_accent(audio_path: str) -> dict:
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    # 格式自动检测，或手动指定 audio/wav / audio/mpeg
    mime = guess_mime(audio_path)

    response = model.generate_content([
        {
            "inline_data": {
                "mime_type": mime,
                "data": audio_b64
            }
        },
        """Please:
1. Transcribe this audio verbatim
2. Identify the speaker's native language background
3. Describe their accent (e.g. Thai, British, Indian Chinese, etc.)
4. Note any pronunciation features: tone, vowel shifts, consonant substitutions
Return JSON: {transcript, language_background, accent_type, notes}"""
    ])
    return response.text

if __name__ == "__main__":
    import glob
    from itertools import chain

    AUDIO_EXTS = ("*.wav", "*.mp3")
    test_files = sorted(
        chain.from_iterable(glob.glob(f"test_audio/{pat}") for pat in AUDIO_EXTS)
    )
    for f in test_files:
        print(f, asr_with_accent(f))