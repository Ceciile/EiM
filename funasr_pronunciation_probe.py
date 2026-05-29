"""Local FunASR pronunciation timing probe.

This script keeps the existing pronunciation `--mode timing` output shape, but
uses local FunASR Paraformer-zh for transcription before applying deterministic
text/pinyin scoring rules.

Example:
  python3 funasr_pronunciation_probe.py \
    --audio "test_audio/sample.wav" \
    --ref "四十是四十，十四是十四。" \
    --mode timing \
    --out funasr_pronunciation_result.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "paraformer-zh"
DEFAULT_NATIVE = "generic"
STRICTNESS_LABELS = {
    1: "strict",
    2: "lenient",
    3: "very_strict",
    4: "very_lenient",
}
STRICTNESS_MULTIPLIER = {
    1: 1.0,
    2: 0.75,
    3: 1.25,
    4: 0.55,
}
PUNCT_RE = re.compile(
    r"[\s\u3000-\u303f\uff00-\uffef"
    r"""!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~"""
    r"""，。！？、；：""''（）【】《》…—·"""
    r"]+"
)
INITIALS = (
    "zh",
    "ch",
    "sh",
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "r",
    "z",
    "c",
    "s",
    "y",
    "w",
)


@dataclass(frozen=True)
class PinyinParts:
    syllable: str
    initial: str
    final: str
    tone: int


def normalize_text(text: str) -> str:
    return PUNCT_RE.sub("", text.strip())


def pinyin_syllables(text: str) -> list[str]:
    from pypinyin import Style, lazy_pinyin

    syllables = lazy_pinyin(normalize_text(text), style=Style.TONE3, neutral_tone_with_five=False)
    return [_ensure_tone_digit(s) for s in syllables]


def _ensure_tone_digit(syllable: str) -> str:
    syllable = syllable.replace("u:", "ü").replace("v", "ü")
    if re.search(r"[0-5]$", syllable):
        return syllable[:-1] + ("0" if syllable[-1] == "5" else syllable[-1])
    return syllable + "0"


def split_pinyin(syllable: str) -> PinyinParts:
    normalized = _ensure_tone_digit(syllable.lower())
    tone = int(normalized[-1]) if normalized[-1].isdigit() else 0
    body = normalized[:-1] if normalized[-1].isdigit() else normalized
    initial = ""
    final = body
    for candidate in INITIALS:
        if body.startswith(candidate):
            initial = candidate
            final = body[len(candidate) :]
            break
    return PinyinParts(syllable=normalized, initial=initial, final=final, tone=tone)


def extract_recognized_text(raw_result: Any) -> tuple[str, Any]:
    """Accept common FunASR result shapes and return (text, timestamp/alignment)."""
    if isinstance(raw_result, list):
        texts: list[str] = []
        timestamps: list[Any] = []
        for item in raw_result:
            if isinstance(item, dict):
                texts.append(str(item.get("text", "")))
                timestamps.extend(item.get("timestamp") or [])
            elif item is not None:
                texts.append(str(item))
        return "".join(texts).strip(), timestamps

    if isinstance(raw_result, dict):
        return str(raw_result.get("text", "")).strip(), raw_result.get("timestamp") or []

    return str(raw_result or "").strip(), []


def load_funasr_model(model_name: str):
    try:
        from funasr import AutoModel
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "FunASR requires PyTorch, but torch is not installed. "
                "Install the CPU build with: python3 -m pip install torch"
            ) from exc
        if exc.name == "torchaudio":
            raise RuntimeError(
                "FunASR requires torchaudio, but it is not installed. "
                "Install the CPU audio stack with: python3 -m pip install torch torchaudio"
            ) from exc
        raise

    try:
        return AutoModel(model=model_name, disable_update=True)
    except TypeError:
        return AutoModel.from_pretrained(model_name)


def transcribe_audio(audio_path: Path, model_name: str) -> tuple[str, Any, Any]:
    model = load_funasr_model(model_name)
    raw_result = model.generate(input=str(audio_path))
    recognized_text, alignment = extract_recognized_text(raw_result)
    return recognized_text, alignment, raw_result


def align_phonemes(reference_text: str, recognized_text: str) -> list[dict[str, Any]]:
    """Placeholder alignment boundary.

    Current implementation is text/pinyin based. A forced-alignment backend can
    replace this function later while keeping scoring/output code stable.
    """
    return build_alignment(reference_text, recognized_text)


def build_alignment(reference_text: str, recognized_text: str) -> list[dict[str, Any]]:
    ref = normalize_text(reference_text)
    hyp = normalize_text(recognized_text)
    ref_pinyin = pinyin_syllables(ref)
    hyp_pinyin = pinyin_syllables(hyp)
    matcher = difflib.SequenceMatcher(a=list(ref), b=list(hyp), autojunk=False)
    rows: list[dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            for i in range(i1, i2):
                rows.append(_alignment_row(i, ref[i], None, ref_pinyin[i], None, ["omission"]))
            continue
        if tag == "insert":
            for offset, j in enumerate(range(j1, j2)):
                rows.append(
                    _alignment_row(
                        i1 + offset,
                        "",
                        hyp[j],
                        None,
                        hyp_pinyin[j],
                        ["insertion"],
                    )
                )
            continue

        span = max(i2 - i1, j2 - j1)
        for offset in range(span):
            i = i1 + offset
            j = j1 + offset
            ref_char = ref[i] if i < i2 else None
            hyp_char = hyp[j] if j < j2 else None
            expected = ref_pinyin[i] if ref_char is not None else None
            observed = hyp_pinyin[j] if hyp_char is not None else None
            if ref_char is None:
                issues = ["insertion"]
            elif hyp_char is None:
                issues = ["omission"]
            else:
                issues = compare_pronunciation(expected, observed)
                if ref_char != hyp_char and "character_diff" not in issues:
                    issues.append("character_diff")
            rows.append(_alignment_row(i, ref_char, hyp_char, expected, observed, issues))

    return rows


def _alignment_row(
    index: int,
    ref_char: str | None,
    hyp_char: str | None,
    expected_pinyin: str | None,
    observed_pinyin: str | None,
    issue_types: list[str],
) -> dict[str, Any]:
    return {
        "index": index,
        "text": ref_char or "",
        "observed_text": hyp_char or "",
        "expected_pinyin": expected_pinyin or "",
        "observed_pronunciation": observed_pinyin or "missing",
        "issue_types": issue_types,
    }


def compare_pronunciation(expected: str | None, observed: str | None) -> list[str]:
    if not expected or not observed:
        return ["omission"] if expected else ["insertion"]
    exp = split_pinyin(expected)
    obs = split_pinyin(observed)
    issues: list[str] = []
    if exp.initial != obs.initial:
        issues.append("initial")
    if exp.final != obs.final:
        issues.append("final")
    if exp.tone != obs.tone:
        issues.append("tone")
    return issues or ["character_diff"]


def score_alignment(rows: list[dict[str, Any]], strictness: int) -> tuple[dict[str, int], list[dict[str, Any]]]:
    multiplier = STRICTNESS_MULTIPLIER[strictness]
    pronunciation_penalty = 0.0
    tone_penalty = 0.0
    fluency_penalty = 0.0
    enriched: list[dict[str, Any]] = []

    for row in rows:
        base = max(issue_penalty(issue) for issue in row["issue_types"])
        penalty = base * multiplier
        if any(issue in row["issue_types"] for issue in ("initial", "final", "character_diff", "omission")):
            pronunciation_penalty += penalty
        if "tone" in row["issue_types"]:
            tone_penalty += penalty
        if any(issue in row["issue_types"] for issue in ("omission", "insertion")):
            fluency_penalty += penalty

        item_score = max(0, round(100 - penalty * 4))
        enriched.append(
            {
                **row,
                "score": item_score,
                "severity": severity_for_penalty(penalty),
                "highlight": True,
            }
        )

    pronunciation_score = clamp_score(100 - pronunciation_penalty)
    tone_score = clamp_score(100 - tone_penalty)
    fluency_score = clamp_score(100 - fluency_penalty)
    overall_score = clamp_score((pronunciation_score * 0.45) + (tone_score * 0.35) + (fluency_score * 0.20))
    return (
        {
            "overall_score": overall_score,
            "pronunciation_score": pronunciation_score,
            "tone_score": tone_score,
            "fluency_score": fluency_score,
        },
        enriched,
    )


def issue_penalty(issue: str) -> int:
    return {
        "omission": 18,
        "insertion": 10,
        "initial": 14,
        "final": 12,
        "tone": 8,
        "character_diff": 10,
    }.get(issue, 6)


def severity_for_penalty(penalty: float) -> str:
    if penalty >= 16:
        return "critical"
    if penalty >= 10:
        return "major"
    return "minor"


def clamp_score(value: float) -> int:
    return max(0, min(100, round(value)))


def priority_error_counts(highlights: list[dict[str, Any]]) -> dict[str, int]:
    retroflex = 0
    umlaut = 0
    tone = 0
    for item in highlights:
        expected = split_pinyin(item["expected_pinyin"]) if item.get("expected_pinyin") else None
        observed = split_pinyin(item["observed_pronunciation"]) if item.get("observed_pronunciation") not in ("", "missing") else None
        issues = set(item.get("issue_types", []))
        if "tone" in issues:
            tone += 1
        if expected and observed and expected.initial in {"zh", "ch", "sh"} and expected.initial != observed.initial:
            retroflex += 1
        if expected and "ü" in expected.final and (not observed or "ü" not in observed.final):
            umlaut += 1
    return {
        "retroflex_zh_ch_sh": retroflex,
        "u_umlaut": umlaut,
        "tone": tone,
    }


def build_timing_result(
    *,
    reference_text: str,
    recognized_text: str,
    native: str,
    strictness: int,
    elapsed_sec: float,
    audio_path: Path,
    model: str,
    asr_alignment: Any | None = None,
) -> dict[str, Any]:
    rows = align_phonemes(reference_text, recognized_text)
    scores, highlights = score_alignment(rows, strictness)
    result = {
        "reference_text": reference_text,
        "recognized_text": recognized_text,
        "native_background": native,
        "strictness": strictness,
        **scores,
        "is_usable_for_word_highlight": True,
        "word_highlights": highlights,
        "priority_error_counts": priority_error_counts(highlights),
    }
    if asr_alignment:
        result["asr_alignment"] = asr_alignment
    return {
        "meta": {
            "audio": str(audio_path),
            "model": model,
            "native": native,
            "strictness": strictness,
            "mode": "timing",
            "elapsed_sec": round(elapsed_sec, 3),
        },
        "result": result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local FunASR Mandarin pronunciation timing probe")
    parser.add_argument("--audio", required=True, help="Path to learner audio file")
    parser.add_argument("--ref", required=True, help="Reference Chinese text")
    parser.add_argument(
        "--native",
        default=DEFAULT_NATIVE,
        choices=["filipino", "generic", "mongolian", "thai", "vietnamese"],
        help="Learner native language background",
    )
    parser.add_argument(
        "--strictness",
        type=int,
        default=2,
        choices=sorted(STRICTNESS_LABELS),
        help="1=strict, 2=lenient, 3=very strict, 4=very lenient",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="FunASR model name or local model path")
    parser.add_argument(
        "--mode",
        default="timing",
        choices=["timing"],
        help="Only timing mode is supported for this local rule-based scorer",
    )
    parser.add_argument("--out", default="funasr_pronunciation_result.json", help="Output JSON path")
    parser.add_argument("--print-raw", action="store_true", help="Print raw FunASR result too")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    recognized_text, asr_alignment, raw_result = transcribe_audio(audio_path, args.model)
    elapsed_sec = time.perf_counter() - t0

    payload = build_timing_result(
        reference_text=args.ref,
        recognized_text=recognized_text,
        native=args.native,
        strictness=args.strictness,
        elapsed_sec=elapsed_sec,
        audio_path=audio_path,
        model=args.model,
        asr_alignment=asr_alignment,
    )
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")
    if args.print_raw:
        print("\n--- raw FunASR result ---")
        print(json.dumps(raw_result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
