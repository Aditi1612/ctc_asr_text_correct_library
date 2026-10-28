"""Engine / corrector configuration and default asset resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Best-effort default: the checkpoint that ships with the sibling service_ctc repo
# (only meaningful on the machine that repo lives on). Off-server, use a HF Hub
# repo (hf_repo=... / CTC_HF_REPO) or explicit model_dir / CTC_MODEL_DIR.
_SIBLING_BUNDLE = Path(__file__).resolve().parents[2] / "service_ctc" / "transcribe_bundle"

# Values used by the serving pipeline (production_pipeline/config.yaml + env defaults).
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_GEMINI_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 512


def default_model_dir() -> Optional[str]:
    env = os.environ.get("CTC_MODEL_DIR")
    if env:
        return env
    candidate = _SIBLING_BUNDLE / "checkpoint"
    return str(candidate) if candidate.exists() else None


def default_vocab_path() -> Optional[str]:
    env = os.environ.get("CTC_VOCAB_PATH")
    if env:
        return env
    candidate = _SIBLING_BUNDLE / "vocab.json"
    return str(candidate) if candidate.exists() else None


def gemini_api_keys_from_env() -> List[str]:
    """``GEMINI_API_KEYS`` (comma-separated) wins over a single ``GEMINI_API_KEY``."""
    multi = os.environ.get("GEMINI_API_KEYS", "")
    if multi.strip():
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.environ.get("GEMINI_API_KEY", "").strip()
    return [single] if single else []


@dataclass(frozen=True)
class EngineConfig:
    """Immutable STT inference configuration.

    Frozen so it can be used as a cache key for a loaded engine.
    """

    model_dir: str
    vocab_path: str
    target_sr: int = 16000
    min_len: int = 1600
    language: str = "korean"
    task: str = "transcribe"
    force_cpu: bool = False
    use_half: bool = False
    # Silence padding applied around each clip before inference (matches serving path).
    pad_left_seconds: float = 0.25
    pad_right_seconds: float = 0.15

    @classmethod
    def resolve(
        cls,
        model_dir: Optional[str] = None,
        vocab_path: Optional[str] = None,
        *,
        hf_repo: Optional[str] = None,
        hf_revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "EngineConfig":
        """Resolve asset locations.

        Priority: explicit ``model_dir``/``vocab_path`` > ``hf_repo`` (arg or
        ``CTC_HF_REPO`` env) > sibling ``service_ctc`` checkpoint.
        """
        repo = hf_repo or os.environ.get("CTC_HF_REPO")

        if model_dir and vocab_path:
            md, vp = model_dir, vocab_path
        elif repo:
            # An explicit hf_repo beats the sibling fallback; a partial explicit
            # model_dir / vocab_path still overrides the matching hub path.
            from .hub import resolve_from_hub

            hub_md, hub_vp = resolve_from_hub(
                repo,
                revision=hf_revision or os.environ.get("CTC_HF_REVISION"),
                token=hf_token,
                cache_dir=hf_cache_dir or os.environ.get("CTC_HF_CACHE_DIR"),
            )
            md = model_dir or hub_md
            vp = vocab_path or hub_vp
        else:
            md = model_dir or default_model_dir()
            vp = vocab_path or default_vocab_path()

        if not md:
            raise ValueError(
                "No checkpoint found. Pass model_dir=..., set CTC_MODEL_DIR, "
                "or pass hf_repo=... / set CTC_HF_REPO."
            )
        if not vp:
            raise ValueError(
                "No vocab.json found. Pass vocab_path=..., set CTC_VOCAB_PATH, "
                "or include vocab.json in the HF repo."
            )
        return cls(model_dir=str(md), vocab_path=str(vp), **kwargs)


@dataclass(frozen=True)
class CorrectorConfig:
    """Gemini transcript-correction settings."""

    api_keys: Tuple[str, ...]
    model: str = DEFAULT_GEMINI_MODEL
    temperature: float = DEFAULT_GEMINI_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    prompt: Optional[str] = None
    # On total failure, return whitespace-cleaned raw text instead of raising
    # (same behavior as the serving pipeline).
    fallback: bool = True

    @classmethod
    def resolve(
        cls,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> "CorrectorConfig":
        """Priority: ``api_keys`` > ``api_key`` > ``GEMINI_API_KEYS`` > ``GEMINI_API_KEY``."""
        keys = list(api_keys or ([api_key] if api_key else gemini_api_keys_from_env()))
        keys = [k for k in keys if k]
        if not keys:
            raise ValueError(
                "No Gemini API key found. Pass api_key=... / api_keys=[...], "
                "or set GEMINI_API_KEY / GEMINI_API_KEYS."
            )
        if "model" not in kwargs and os.environ.get("GEMINI_MODEL"):
            kwargs["model"] = os.environ["GEMINI_MODEL"]
        return cls(api_keys=tuple(keys), **kwargs)
