"""
ASR 性能验证：单次响应耗时 + 与参考文本的 CER / WER。

参考文本放在 test_audio/*.txt，音频与 txt 的对应关系见 test_audio/manifest.json。
"""

from __future__ import annotations

import json
import re
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from asr_spike import asr_with_accent, load_env_file, ENV_FILE

TEST_AUDIO_DIR = Path(__file__).resolve().parent / "test_audio"
MANIFEST_PATH = TEST_AUDIO_DIR / "manifest.json"
RESULTS_PATH = Path(__file__).resolve().parent / "asr_eval_results.json"

# 去掉标点、空白，便于与 ASR 输出对齐比较
_PUNCT_RE = re.compile(
    r"[\s\u3000-\u303f\uff00-\uffef"
    r"""!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~"""
    r"""，。！？、；：""''（）【】《》…—·"""
    r"]+"
)


def normalize(text: str) -> str:
    return _PUNCT_RE.sub("", text.strip())


def _levenshtein_seq(ref: list, hyp: list) -> int:
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        curr = [i]
        for j, hc in enumerate(hyp, 1):
            cost = 0 if rc == hc else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def levenshtein(ref: str, hyp: str) -> int:
    return _levenshtein_seq(list(ref), list(hyp))


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate = 编辑距离 / 参考字符数"""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def wer(ref: str, hyp: str) -> float:
    """
    Word Error Rate：中文默认用 jieba 分词；未安装时退化为按字分词（与 CER 数值相同）。
    """
    r, h = normalize(ref), normalize(hyp)
    try:
        import jieba

        r_words = list(jieba.cut(r, cut_all=False))
        h_words = list(jieba.cut(h, cut_all=False))
        r_joined = " ".join(w for w in r_words if w)
        h_joined = " ".join(w for w in h_words if w)
    except ImportError:
        r_joined = " ".join(r)
        h_joined = " ".join(h)
    ref_words = r_joined.split()
    hyp_words = h_joined.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein_seq(ref_words, hyp_words) / len(ref_words)


def extract_transcript(raw: str) -> str:
    """从模型返回中取出 transcript 字段；失败则返回原文本。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("transcript"):
            return str(data["transcript"])
    except json.JSONDecodeError:
        pass
    m = re.search(r'"transcript"\s*:\s*"((?:\\.|[^"\\])*)"', raw, re.DOTALL)
    if m:
        return json.loads(f'"{m.group(1)}"')
    return raw


def load_manifest() -> dict[str, str]:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {audio: ref for audio, ref in data.items()}


def load_reference(filename: str) -> str:
    return (TEST_AUDIO_DIR / filename).read_text(encoding="utf-8").strip()


def audio_duration_sec(path: Path) -> float | None:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    try:
        from mutagen import File as MutagenFile

        info = MutagenFile(path)
        if info and info.info and getattr(info.info, "length", None):
            return float(info.info.length)
    except Exception:
        pass
    return None


@dataclass
class EvalRow:
    audio: str
    reference_file: str
    latency_sec: float
    audio_duration_sec: float | None
    rtf: float | None
    cer: float
    wer: float
    reference_chars: int
    hypothesis_chars: int
    reference_preview: str
    hypothesis_preview: str
    raw_response_preview: str


def evaluate_one(audio_name: str, ref_filename: str) -> EvalRow:
    audio_path = TEST_AUDIO_DIR / audio_name
    reference = load_reference(ref_filename)

    t0 = time.perf_counter()
    raw = asr_with_accent(str(audio_path))
    latency = time.perf_counter() - t0

    hypothesis = extract_transcript(raw)
    duration = audio_duration_sec(audio_path)
    rtf = round(latency / duration, 3) if duration and duration > 0 else None
    return EvalRow(
        audio=audio_name,
        reference_file=ref_filename,
        latency_sec=round(latency, 3),
        audio_duration_sec=round(duration, 3) if duration else None,
        rtf=rtf,
        cer=round(cer(reference, hypothesis), 4),
        wer=round(wer(reference, hypothesis), 4),
        reference_chars=len(normalize(reference)),
        hypothesis_chars=len(normalize(hypothesis)),
        reference_preview=reference[:80] + ("…" if len(reference) > 80 else ""),
        hypothesis_preview=hypothesis[:80] + ("…" if len(hypothesis) > 80 else ""),
        raw_response_preview=raw[:200] + ("…" if len(raw) > 200 else ""),
    )


def main() -> None:
    load_env_file(ENV_FILE)
    manifest = load_manifest()
    rows = [evaluate_one(audio, ref) for audio, ref in manifest.items()]

    print("\n=== ASR 评估结果 ===\n")
    print(
        f"{'音频':<28} {'耗时(s)':>8} {'RTF':>8} {'CER':>8} {'WER':>8}  参考文本"
    )
    print("-" * 80)
    for r in rows:
        rtf_s = f"{r.rtf:>8.3f}" if r.rtf is not None else "     n/a"
        print(
            f"{r.audio:<28} {r.latency_sec:>8.3f} {rtf_s} {r.cer:>8.2%} {r.wer:>8.2%}  {r.reference_file}"
        )

    summary = {
        "count": len(rows),
        "latency_sec_avg": round(sum(r.latency_sec for r in rows) / len(rows), 3) if rows else 0,
        "cer_avg": round(sum(r.cer for r in rows) / len(rows), 4) if rows else 0,
        "wer_avg": round(sum(r.wer for r in rows) / len(rows), 4) if rows else 0,
        "rows": [asdict(r) for r in rows],
    }
    RESULTS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细结果已写入: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
