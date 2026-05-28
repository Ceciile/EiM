"""Ali Omni pronunciation scoring spike.

Standalone script only: it does not modify or import the existing ASR scripts.

Example:
  python3 ali_pronunciation_probe.py \
    --audio "test_audio/sample.mp3" \
    --ref "四十是四十，十四是十四，绿女去学习。" \
    --native thai \
    --strictness 1 \
    --out ali_pronunciation_result.json
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI


ENV_FILE = Path(__file__).resolve().parent / "env"
DEFAULT_MODEL = "qwen3.5-omni-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

STRICTNESS_LABELS = {
    3: "very_strict",
    1: "strict",
    2: "lenient",
    4: "very_lenient",
}

NATIVE_BACKGROUND_RULES = {
    "thai": {
        "tone_policy": "Medium strictness. Thai has tones, so tone awareness exists, but Mandarin tone contours still need correction.",
        "critical_items": ["zh/ch/sh retroflex initials", "ü rounded front vowel", "tone contour"],
    },
    "filipino": {
        "tone_policy": "User-friendly tone scoring. Filipino has no lexical tone, so tone mistakes should be highlighted clearly but not over-penalized.",
        "critical_items": ["zh/ch/sh retroflex initials", "ü rounded front vowel", "tone contour"],
    },
    "vietnamese": {
        "tone_policy": "Relatively strict tone scoring. Vietnamese has six tones, so transfer is expected to be stronger.",
        "critical_items": ["zh/ch/sh retroflex initials", "ü rounded front vowel", "final nasal -n/-ng"],
    },
    "mongolian": {
        "tone_policy": "User-friendly tone scoring. Mongolian has no lexical tone, so tone errors should receive graduated feedback.",
        "critical_items": ["zh/ch/sh retroflex initials", "ü rounded front vowel", "tone contour"],
    },
    "generic": {
        "tone_policy": "Balanced scoring.",
        "critical_items": ["zh/ch/sh retroflex initials", "ü rounded front vowel", "tone contour"],
    },
}


def load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    aliases = {
        "alibabacloud_api_key": "DASHSCOPE_API_KEY",
        "ALIBABACLOUD_API_KEY": "DASHSCOPE_API_KEY",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        os.environ.setdefault(key, value)
        if key in aliases:
            os.environ.setdefault(aliases[key], value)


def dashscope_client() -> OpenAI:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("alibabacloud_api_key")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY or alibabacloud_api_key in env")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def guess_mime(audio_path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(audio_path))
    if guessed:
        return guessed
    ext = audio_path.suffix.lower()
    if ext == ".wav":
        return "audio/wav"
    if ext in {".mp3", ".mpeg"}:
        return "audio/mpeg"
    if ext in {".m4a", ".mp4"}:
        return "audio/mp4"
    return "application/octet-stream"


def audio_data_uri(audio_path: Path) -> str:
    data = audio_path.read_bytes()
    mime = guess_mime(audio_path)
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def strictness_instruction(strictness: int) -> str:
    label = STRICTNESS_LABELS[strictness]
    return {
        1: "Strict: apply normal correction pressure. Penalize clear initial/final/tone errors.",
        2: "Lenient: be friendly to low-score users. Highlight errors, but reduce penalty for understandable pronunciation.",
        3: "Very strict: strongly penalize zh/ch/sh, ü, and tone mistakes; use this for diagnostic drills.",
        4: "Very lenient: prioritize encouragement. Penalize only errors that block comprehension.",
    }[strictness] + f" Internal label: {label}."


def build_prompt(ref_text: str, native: str, strictness: int) -> str:
    rules = NATIVE_BACKGROUND_RULES.get(native, NATIVE_BACKGROUND_RULES["generic"])
    critical_items = "、".join(rules["critical_items"])
    return f"""You are a Mandarin pronunciation assessment engine for language learners.

Task:
Assess the learner audio against this reference text:
{ref_text}

Return ONLY one valid JSON object. Do not include markdown.

Scoring policy:
- strictness={strictness}: {strictness_instruction(strictness)}
- Native background: {native}
- Tone policy: {rules["tone_policy"]}
- Critical items to prioritize: {critical_items}
- zh/ch/sh retroflex initials are strict items for Thai, Filipino, Vietnamese, and Mongolian learners.
- ü (rounded front vowel) is strict for Thai and Filipino learners, and still important for Vietnamese and Mongolian learners.
- Tone scoring must be graduated: Vietnamese can be stricter, Thai medium, Filipino/Mongolian more user-friendly.

Required JSON schema:
{{
  "reference_text": "...",
  "recognized_text": "...",
  "native_background": "{native}",
  "strictness": {strictness},
  "overall_score": 0,
  "pronunciation_score": 0,
  "tone_score": 0,
  "fluency_score": 0,
  "is_usable_for_word_highlight": true,
  "summary": "...",
  "word_highlights": [
    {{
      "index": 0,
      "text": "字 or word from reference",
      "expected_pinyin": "shi4",
      "observed_pronunciation": "si4 or unclear",
      "score": 0,
      "severity": "ok | minor | major | critical",
      "highlight": false,
      "issue_types": ["initial", "final", "tone", "omission", "insertion", "fluency"],
      "initial": {{"expected": "sh", "observed": "s", "score": 0, "comment": "..."}},
      "final": {{"expected": "i", "observed": "i", "score": 0, "comment": "..."}},
      "tone": {{"expected": 4, "observed": 4, "score": 0, "comment": "..."}},
      "suggestion": "short learner-facing correction"
    }}
  ],
  "priority_errors": [
    {{"type": "retroflex_zh_ch_sh", "count": 0, "examples": ["..."], "teaching_tip": "..."}},
    {{"type": "u_umlaut", "count": 0, "examples": ["..."], "teaching_tip": "..."}},
    {{"type": "tone", "count": 0, "examples": ["..."], "teaching_tip": "..."}}
  ],
  "learner_friendly_feedback": "..."
}}

Important:
- If exact acoustic phoneme detection is uncertain, say so in the comment, but still provide the best estimate.
- For word_highlights, prefer Chinese character-level highlights for Chinese reference text.
- A high score means the issue has smaller negative impact; a low score means the issue is more serious.
"""


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def score_pronunciation(
    audio_path: Path,
    ref_text: str,
    native: str,
    strictness: int,
    model: str,
) -> tuple[dict[str, Any], str]:
    client = dashscope_client()
    prompt = build_prompt(ref_text=ref_text, native=native, strictness=strictness)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": audio_data_uri(audio_path)}},
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    return extract_json(raw), raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ali Omni Mandarin pronunciation scoring probe")
    parser.add_argument("--audio", required=True, help="Path to learner audio file")
    parser.add_argument("--ref", required=True, help="Reference Chinese text")
    parser.add_argument(
        "--native",
        default="generic",
        choices=sorted(NATIVE_BACKGROUND_RULES),
        help="Learner native language background",
    )
    parser.add_argument(
        "--strictness",
        type=int,
        default=2,
        choices=sorted(STRICTNESS_LABELS),
        help="1=strict, 2=lenient, 3=very strict, 4=very lenient",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DASHSCOPE_OMNI_MODEL", DEFAULT_MODEL),
        help="DashScope Omni model name",
    )
    parser.add_argument("--out", default="ali_pronunciation_result.json", help="Output JSON path")
    parser.add_argument("--print-raw", action="store_true", help="Print raw model response too")
    return parser.parse_args()


def main() -> int:
    load_env_file()
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 2

    result, raw = score_pronunciation(
        audio_path=audio_path,
        ref_text=args.ref,
        native=args.native,
        strictness=args.strictness,
        model=args.model,
    )

    payload = {
        "meta": {
            "audio": str(audio_path),
            "model": args.model,
            "native": args.native,
            "strictness": args.strictness,
        },
        "result": result,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")
    if args.print_raw:
        print("\n--- raw response ---")
        print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
