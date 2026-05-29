"""Persistent HTTP server for local FunASR pronunciation timing scoring.

The CLI probe is useful for debugging, but each CLI invocation pays Python,
PyTorch, and model-loading startup cost. This server loads the FunASR model once
at process startup and reuses it for every `/score` request.

Example:
  python3 funasr_pronunciation_server.py --model paraformer-zh --host 127.0.0.1 --port 8008

Then score a local file:
  curl -X POST http://127.0.0.1:8008/score \
    -H 'Content-Type: application/json' \
    -d '{"audio_path":"test_audio/sample.wav","ref":"四十是四十","native":"thai","strictness":2}'
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from funasr_pronunciation_probe import (
    DEFAULT_MODEL,
    build_timing_result,
    extract_recognized_text,
    load_funasr_model,
)

ModelLoader = Callable[[str], Any]
QWEN_API_MODEL = "qwen3.5-omni-plus"
DEFAULT_FUNASR_MODEL = "iic/SenseVoiceSmall"
PROVIDER_MODELS = {
    "funasr": ["iic/SenseVoiceSmall", "paraformer-zh"],
    "qwenAPI": [QWEN_API_MODEL],
}


class PersistentFunasrScorer:
    """Keeps one FunASR model instance warm for repeated scoring requests."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        model_loader: ModelLoader = load_funasr_model,
    ) -> None:
        self.model_name = model_name
        self._model_loader = model_loader
        self._model: Any | None = None
        self._lock = Lock()

    def warmup(self) -> None:
        self._get_model()

    def score_audio_path(
        self,
        *,
        audio_path: Path,
        reference_text: str,
        native: str,
        strictness: int,
    ) -> dict[str, Any]:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        t0 = time.perf_counter()
        model = self._get_model()
        # First version is deliberately serialized; multiple workers can be
        # started later if throughput matters more than RAM usage.
        with self._lock:
            raw_result = model.generate(input=str(audio_path))
        recognized_text, asr_alignment = extract_recognized_text(raw_result)
        elapsed_sec = time.perf_counter() - t0

        return build_timing_result(
            reference_text=reference_text,
            recognized_text=recognized_text,
            native=native,
            strictness=strictness,
            elapsed_sec=elapsed_sec,
            audio_path=audio_path,
            model=self.model_name,
            asr_alignment=asr_alignment,
        )

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._model_loader(self.model_name)
        return self._model


def score_with_qwen_api(
    *,
    audio_path: Path,
    reference_text: str,
    native: str,
    strictness: int,
    model: str = QWEN_API_MODEL,
) -> dict[str, Any]:
    from ali_pronunciation_probe import build_output_payload, load_env_file, score_pronunciation

    load_env_file()
    t0 = time.perf_counter()
    result, _raw = score_pronunciation(
        audio_path=audio_path,
        ref_text=reference_text,
        native=native,
        strictness=strictness,
        model=model,
        mode="timing",
    )
    payload = build_output_payload(
        audio_path=audio_path,
        model=model,
        native=native,
        strictness=strictness,
        mode="timing",
        elapsed_sec=time.perf_counter() - t0,
        result=result,
    )
    payload["meta"]["provider"] = "qwenAPI"
    return payload


def create_app(
    scorer: PersistentFunasrScorer | None = None,
    funasr_scorer_factory: Callable[[str], Any] = PersistentFunasrScorer,
    qwen_scorer: Callable[..., dict[str, Any]] = score_with_qwen_api,
):
    app = FastAPI(title="Local FunASR Pronunciation Server")
    default_scorer = scorer or create_funasr_scorer(funasr_scorer_factory, DEFAULT_FUNASR_MODEL)
    default_funasr_model = default_scorer.model_name
    funasr_scorers: dict[str, Any] = {default_scorer.model_name: default_scorer}

    def get_funasr_scorer(model_name: str) -> Any:
        if model_name not in funasr_scorers:
            funasr_scorers[model_name] = create_funasr_scorer(funasr_scorer_factory, model_name)
        return funasr_scorers[model_name]

    class ScoreRequest(BaseModel):
        audio_path: str = Field(..., description="Path to an audio file accessible to this server")
        ref: str = Field(..., description="Reference Chinese text")
        native: str = Field("generic", description="Learner native language background")
        strictness: int = Field(2, ge=1, le=4, description="1=strict, 2=lenient, 3=very strict, 4=very lenient")
        provider: str = Field("funasr", description="funasr | qwenAPI")
        model: Optional[str] = Field(None, description="Optional backend model override")

    @app.on_event("startup")
    def _startup() -> None:
        default_scorer.warmup()

    @app.get("/")
    def index() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "Local FunASR Pronunciation Server",
            "providers": PROVIDER_MODELS,
            "endpoints": {
                "health": "GET /health",
                "score_local_file": "POST /score",
                "score_upload": "POST /score-upload",
            },
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "providers": PROVIDER_MODELS,
            "loaded_funasr_models": sorted(funasr_scorers),
        }

    @app.post("/score")
    def score(request: ScoreRequest) -> dict[str, Any]:
        try:
            return score_by_provider(
                get_funasr_scorer=get_funasr_scorer,
                qwen_scorer=qwen_scorer,
                default_funasr_model=default_funasr_model,
                provider=request.provider,
                model=request.model,
                audio_path=Path(request.audio_path),
                reference_text=request.ref,
                native=request.native,
                strictness=request.strictness,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.post("/score-upload")
    async def score_upload(
        ref: str = Form(...),
        native: str = Form("generic"),
        strictness: int = Form(2),
        provider: str = Form("funasr"),
        model: Optional[str] = Form(None),
        audio: UploadFile = File(...),
    ) -> dict[str, Any]:
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(await audio.read())
            tmp.flush()
            try:
                return score_by_provider(
                    get_funasr_scorer=get_funasr_scorer,
                    qwen_scorer=qwen_scorer,
                    default_funasr_model=default_funasr_model,
                    provider=provider,
                    model=model,
                    audio_path=Path(tmp.name),
                    reference_text=ref,
                    native=native,
                    strictness=strictness,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return app


def create_funasr_scorer(factory: Callable[[str], Any], model_name: str) -> Any:
    try:
        return factory(model_name=model_name)
    except TypeError as keyword_error:
        try:
            return factory(model_name)
        except TypeError:
            raise keyword_error


def score_by_provider(
    *,
    get_funasr_scorer: Callable[[str], Any],
    qwen_scorer: Callable[..., dict[str, Any]],
    default_funasr_model: str,
    provider: str,
    model: str | None,
    audio_path: Path,
    reference_text: str,
    native: str,
    strictness: int,
) -> dict[str, Any]:
    validate_provider_model(provider=provider, model=model)
    if provider == "funasr":
        selected_model = model or default_funasr_model
        active_scorer = get_funasr_scorer(selected_model)
        payload = active_scorer.score_audio_path(
            audio_path=audio_path,
            reference_text=reference_text,
            native=native,
            strictness=strictness,
        )
        payload["meta"]["provider"] = "funasr"
        payload["meta"]["model"] = selected_model
        return payload
    if provider == "qwenAPI":
        return qwen_scorer(
            audio_path=audio_path,
            reference_text=reference_text,
            native=native,
            strictness=strictness,
            model=model or QWEN_API_MODEL,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def validate_provider_model(*, provider: str, model: str | None) -> None:
    if provider not in PROVIDER_MODELS:
        raise ValueError(f"Unsupported provider: {provider}")
    if model is None:
        return
    if model not in PROVIDER_MODELS[provider]:
        allowed = ", ".join(PROVIDER_MODELS[provider])
        raise ValueError(f"Model {model!r} is not valid for provider {provider!r}. Allowed: {allowed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent local FunASR pronunciation scoring server")
    parser.add_argument("--model", default=DEFAULT_FUNASR_MODEL, choices=PROVIDER_MODELS["funasr"], help="Initial FunASR model to warm")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8008, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload for local development")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import uvicorn

    scorer = PersistentFunasrScorer(model_name=args.model)
    app = create_app(scorer)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
