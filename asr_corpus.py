"""测试语料：manifest.json（内联 gt 或 txt）+ 可选 ground_truth.json。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 统一数据目录：test1（旧中文语料）| test_audio（ElevenLabs 等）
_active_data_dir: Path | None = None


def get_data_dir() -> Path:
    if _active_data_dir is not None:
        return _active_data_dir
    name = os.environ.get("ASR_DATA_DIR", "test_audio")
    return ROOT / name


def set_data_dir(path: str | Path) -> Path:
    """切换语料根目录（test1 / test_audio），供 asr_eval / asr_spike 共用。"""
    global _active_data_dir
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {p}")
    _active_data_dir = p
    return p


def manifest_path() -> Path:
    return get_data_dir() / "manifest.json"


def durations_path() -> Path:
    return get_data_dir() / "durations.json"


def ground_truth_path() -> Path:
    return get_data_dir() / "ground_truth.json"


@dataclass(frozen=True)
class CorpusEntry:
    """单条评测/探测语料。"""

    key: str
    path: Path
    gt: str
    note: str = ""
    source: str = "manifest"  # manifest | generated
    ref_file: str = ""
    data_dir: str = ""


def resolve_audio_path(file_field: str) -> Path:
    data_dir = get_data_dir()
    p = Path(file_field)
    if p.is_absolute():
        return p
    for base in (ROOT, data_dir):
        candidate = (base / p).resolve()
        if candidate.is_file():
            return candidate
    return (data_dir / p.name).resolve()


def _resolve_manifest_reference(value: str) -> tuple[str, str]:
    value = value.strip()
    if not value:
        return "", value

    data_dir = get_data_dir()
    if value.endswith(".txt"):
        path = data_dir / value
        if path.is_file():
            return path.read_text(encoding="utf-8").strip(), value

    if len(value) > 30 or " " in value:
        return value, "manifest.json (inline)"

    path = data_dir / value
    if path.is_file() and path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8").strip(), value

    return value, "manifest.json (inline)"


def load_manifest_entries() -> list[CorpusEntry]:
    mp = manifest_path()
    if not mp.exists():
        return []
    data = json.loads(mp.read_text(encoding="utf-8"))
    data_dir = get_data_dir()
    entries: list[CorpusEntry] = []
    for audio_name, ref_value in data.items():
        if audio_name.startswith("_"):
            continue
        path = data_dir / audio_name
        if not path.is_file():
            continue
        gt, ref_label = _resolve_manifest_reference(ref_value)
        note = audio_name
        if "_" in audio_name:
            parts = audio_name.rsplit("_", 2)
            if len(parts) >= 2:
                note = parts[-2] if parts[-1].endswith((".mp3", ".wav")) else audio_name
        entries.append(
            CorpusEntry(
                key=audio_name,
                path=path,
                gt=gt,
                note=note,
                source="manifest",
                ref_file=ref_label,
                data_dir=data_dir.name,
            )
        )
    return entries


def load_ground_truth() -> dict[str, dict]:
    gp = ground_truth_path()
    if not gp.exists():
        return {}
    data = json.loads(gp.read_text(encoding="utf-8"))
    files: dict[str, dict] = {}
    for key, val in data.items():
        files[key] = {
            "path": resolve_audio_path(val["file"]),
            "gt": val["gt"],
            "note": val.get("note", ""),
        }
    return files


def load_manual_durations() -> dict[str, float]:
    dp = durations_path()
    if not dp.exists():
        return {}
    data = json.loads(dp.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in data.items() if not k.startswith("_")}


def load_files_config() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for e in load_manifest_entries():
        out[e.key] = {"path": e.path, "gt": e.gt, "note": e.note}
    for key, cfg in load_ground_truth().items():
        out.setdefault(key, cfg)
    return out


def load_corpus(corpus: str = "manifest") -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    seen: set[str] = set()
    data_dir = get_data_dir()

    if corpus in ("manifest", "all", "legacy"):
        for e in load_manifest_entries():
            entries.append(e)
            seen.add(e.key)

    if corpus in ("generated", "all"):
        for key, cfg in load_ground_truth().items():
            if key in seen:
                continue
            path = cfg["path"]
            if not path.is_file():
                raise FileNotFoundError(
                    f"生成语料缺失: {path}（可改用 --corpus manifest）"
                )
            entries.append(
                CorpusEntry(
                    key=key,
                    path=path,
                    gt=cfg["gt"],
                    note=cfg["note"],
                    source="generated",
                    data_dir=data_dir.name,
                )
            )

    return entries


def iter_existing_paths(corpus: str = "manifest") -> list[Path]:
    return [e.path for e in load_corpus(corpus)]
