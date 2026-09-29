"""Utilities for stable model identifiers used by on-disk caches."""

from typing import Any


# These are the model identifiers supported by this repository's evaluation
# CLI and by the published moment-statistics layout.
_CANONICAL_MODEL_NAMES = (
    "EleutherAI/gpt-j-6B",
    "gpt2-xl",
    "meta-llama/Meta-Llama-3-8B",
)


def canonical_stats_model_name(name_or_path: Any) -> str:
    """Return a stable directory name for cached layer statistics.

    Local loading changes ``config._name_or_path`` from a Hugging Face model ID
    into an absolute path. If that path contains a supported model ID, retain
    the canonical ID so local and hub loading reuse the same statistics.
    Unknown names keep the repository's previous slash-to-underscore behavior.
    """

    normalized = str(name_or_path).replace("\\", "/").rstrip("/")
    parts = [part for part in normalized.split("/") if part]
    lower_parts = [part.lower() for part in parts]

    for canonical_name in _CANONICAL_MODEL_NAMES:
        canonical_parts = canonical_name.split("/")
        lower_canonical_parts = [part.lower() for part in canonical_parts]
        width = len(canonical_parts)
        for start in range(len(parts) - width + 1):
            if lower_parts[start : start + width] == lower_canonical_parts:
                return canonical_name.replace("/", "_")

        # Also recognize Hugging Face's local cache directory convention,
        # e.g. models--EleutherAI--gpt-j-6B.
        hub_cache_component = "models--" + "--".join(canonical_parts)
        if hub_cache_component.lower() in lower_parts:
            return canonical_name.replace("/", "_")

    return normalized.replace("/", "_")
