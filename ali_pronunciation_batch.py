"""Batch runner for Ali Omni pronunciation scoring.

This script is intentionally standalone and does not modify existing ASR scripts.

Default behavior:
  - Read `test_audio/manifest_pro.json`
  - Score audio files whose filename starts with "泰"
  - Write one JSON file per audio into `shadow/`
  - Print nothing on success

Example:
  python3 ali_pronunciation_batch.py

  python3 ali_pronunciation_batch.py \
    --data-dir test_audio \
    --manifest test_audio/manifest_pro.json \
    --prefix 泰 \
    --native thai \
    --strictness 2 \
    --shadow shadow
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ali_pronunciation_probe import DEFAULT_MODEL, load_env_file, score_pronunciation


ScoreResult = tuple[dict[str, Any], str]
Scorer = Callable[[Path, str, str, int, str], ScoreResult]
Clock = Callable[[], float]


def load_manifest(manifest_path: Path) -> dict[str, str]:
    raw = manifest_path.read_text(encoding="utf-8")
    data = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
    return {str(name): str(ref) for name, ref in data.items()}


def safe_output_stem(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r'[\\/:*?"<>|]+', "_", stem).strip() or "audio"


def iter_manifest_entries(
    data_dir: Path,
    manifest: dict[str, str],
    prefix: str,
) -> list[tuple[Path, str, str]]:
    entries: list[tuple[Path, str, str]] = []
    for filename, reference_text in manifest.items():
        if prefix and not filename.startswith(prefix):
            continue
        entries.append((data_dir / filename, filename, reference_text))
    return entries


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def score_batch(
    *,
    data_dir: Path,
    manifest_path: Path,
    shadow_dir: Path,
    prefix: str,
    native: str,
    strictness: int,
    model: str,
    scorer: Scorer = score_pronunciation,
    clock: Clock = time.perf_counter,
) -> list[Path]:
    manifest = load_manifest(manifest_path)
    outputs: list[Path] = []

    for audio_path, filename, reference_text in iter_manifest_entries(data_dir, manifest, prefix):
        out_path = shadow_dir / f"pronunciation_{safe_output_stem(filename)}.json"
        started_at = datetime.now(timezone.utc).isoformat()
        t0 = clock()
        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            result, raw = scorer(audio_path, reference_text, native, strictness, model)
            elapsed = round(clock() - t0, 3)
            payload = {
                "status": "ok",
                "meta": {
                    "audio": str(audio_path),
                    "filename": filename,
                    "reference_text": reference_text,
                    "native": native,
                    "strictness": strictness,
                    "model": model,
                    "started_at": started_at,
                    "elapsed_sec": elapsed,
                },
                "result": result,
                "raw_response": raw,
            }
        except Exception as exc:
            elapsed = round(clock() - t0, 3)
            payload = {
                "status": "error",
                "meta": {
                    "audio": str(audio_path),
                    "filename": filename,
                    "reference_text": reference_text,
                    "native": native,
                    "strictness": strictness,
                    "model": model,
                    "started_at": started_at,
                    "elapsed_sec": elapsed,
                },
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

        write_json(out_path, payload)
        outputs.append(out_path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Ali Omni pronunciation scoring")
    parser.add_argument("--data-dir", default="test_audio", help="Audio directory")
    parser.add_argument(
        "--manifest",
        default="test_audio/manifest_pro.json",
        help="JSON mapping from audio filename to reference text",
    )
    parser.add_argument("--prefix", default="泰", help="Only score files whose filename starts with this prefix")
    parser.add_argument("--shadow", default="shadow", help="Output directory for per-audio JSON files")
    parser.add_argument(
        "--native",
        default="thai",
        choices=["filipino", "generic", "mongolian", "thai", "vietnamese"],
        help="Learner native language background",
    )
    parser.add_argument(
        "--strictness",
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help="1=strict, 2=lenient, 3=very strict, 4=very lenient",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DashScope Omni model name")
    parser.add_argument("--verbose", action="store_true", help="Print output paths")
    return parser.parse_args()


def main() -> int:
    load_env_file()
    args = parse_args()
    outputs = score_batch(
        data_dir=Path(args.data_dir),
        manifest_path=Path(args.manifest),
        shadow_dir=Path(args.shadow),
        prefix=args.prefix,
        native=args.native,
        strictness=args.strictness,
        model=args.model,
    )
    if args.verbose:
        for path in outputs:
            print(path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
