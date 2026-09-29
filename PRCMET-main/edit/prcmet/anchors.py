"""Request-conditioned pre-edit anchor covariances for single-shot PRCMET."""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import uuid

import torch
from tqdm.auto import tqdm

from rome import repr_tools
from util.model_name import canonical_stats_model_name


ANCHOR_COVARIANCE_CONVENTION = "neutral_subject_anchor_raw_ppT_scaled_v2"
QA_ANCHOR_COVARIANCE_CONVENTION = (
    "neutral_held_out_qa_prompt_end_raw_ppT_edit_mass_v2"
)
NEUTRAL_TEMPLATE_CONTEXT_MODE = "neutral_templates"
HELD_OUT_QA_CONTEXT_MODE = "zsre_held_out_qa"
RELATION_ANCHOR_COVARIANCE_CONVENTION = (
    "relation_conditioned_hard_anchor_uniform_budget_v3"
)
RELATION_ANCHOR_HARDNESS_METHOD = "absolute_cosine_to_normalized_relation_center"
RELATION_ANCHOR_BUDGET_METHOD = "edit_frequency_capacity_largest_remainder_v1"
DEFAULT_ANCHOR_TEMPLATES = (
    "{}",
    "This text is about {}.",
    "A short description of {}.",
)


@dataclass(frozen=True)
class AnchorConfig:
    templates: Sequence[str]
    max_count: int
    extraction_chunk_size: int
    covariance_dtype: str

    @classmethod
    def from_hparams(
        cls, hparams: Any, context_mode: str = NEUTRAL_TEMPLATE_CONTEXT_MODE
    ) -> "AnchorConfig":
        # QA contexts come exclusively from held-out questions. Do not consult
        # or temporarily replace hparams.anchor_templates in this mode.
        templates = (
            ()
            if context_mode == HELD_OUT_QA_CONTEXT_MODE
            else getattr(hparams, "anchor_templates", None)
        )
        config = cls(
            templates=(
                tuple(DEFAULT_ANCHOR_TEMPLATES)
                if templates is None
                else tuple(templates)
            ),
            max_count=getattr(hparams, "anchor_max_count", 10000),
            extraction_chunk_size=getattr(
                hparams, "anchor_extraction_chunk_size", 128
            ),
            covariance_dtype=getattr(
                hparams, "anchor_covariance_dtype", "float32"
            ),
        )
        config.validate(context_mode)
        return config

    def validate(self, context_mode: str = NEUTRAL_TEMPLATE_CONTEXT_MODE) -> None:
        if context_mode not in {NEUTRAL_TEMPLATE_CONTEXT_MODE, HELD_OUT_QA_CONTEXT_MODE}:
            raise ValueError(f"Unknown neutral anchor context mode: {context_mode}")
        if context_mode == NEUTRAL_TEMPLATE_CONTEXT_MODE:
            if not self.templates or not all(isinstance(x, str) for x in self.templates):
                raise ValueError("anchor_templates must be a non-empty list of strings.")
            if not all(template.count("{}") == 1 for template in self.templates):
                raise ValueError("Every anchor template must contain exactly one '{}'.")
        for name, value in (
            ("anchor_max_count", self.max_count),
            ("anchor_extraction_chunk_size", self.extraction_chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.covariance_dtype not in {"float32", "float64"}:
            raise ValueError("anchor_covariance_dtype must be 'float32' or 'float64'.")


@dataclass(frozen=True)
class AnchorCovarianceArtifact:
    path: Path
    candidate_count: int
    anchor_count: int
    scale: float
    fingerprint: str
    convention: str = ANCHOR_COVARIANCE_CONVENTION


@dataclass(frozen=True)
class RelationAnchorConfig:
    enabled: bool
    beta: float
    budget: int
    max_count: int
    extraction_chunk_size: int
    covariance_dtype: str

    @classmethod
    def from_hparams(cls, hparams: Any) -> "RelationAnchorConfig":
        config = cls(
            enabled=getattr(hparams, "relation_anchor_preservation", False),
            beta=getattr(hparams, "relation_anchor_beta", 0.15),
            budget=getattr(hparams, "relation_anchor_budget", 1024),
            max_count=getattr(hparams, "anchor_max_count", 10000),
            extraction_chunk_size=getattr(
                hparams, "anchor_extraction_chunk_size", 128
            ),
            covariance_dtype=getattr(
                hparams, "anchor_covariance_dtype", "float32"
            ),
        )
        config.validate(
            preservation_constraint=getattr(
                hparams, "preservation_constraint", False
            ),
            preservation_mode=getattr(hparams, "preservation_mode", "history"),
        )
        return config

    def validate(
        self, preservation_constraint: bool, preservation_mode: str
    ) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("relation_anchor_preservation must be a bool.")
        if isinstance(self.beta, bool) or not isinstance(self.beta, (int, float)):
            raise ValueError("relation_anchor_beta must be a finite number >= 0.")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("relation_anchor_beta must be a finite number >= 0.")
        for name, value in (
            ("relation_anchor_budget", self.budget),
            ("anchor_max_count", self.max_count),
            ("anchor_extraction_chunk_size", self.extraction_chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.covariance_dtype not in {"float32", "float64"}:
            raise ValueError("anchor_covariance_dtype must be 'float32' or 'float64'.")
        if self.enabled:
            if preservation_constraint is not True:
                raise ValueError(
                    "relation_anchor_preservation=true requires "
                    "preservation_constraint=true."
                )
            if preservation_mode != "anchors":
                raise ValueError(
                    "relation_anchor_preservation=true requires "
                    "preservation_mode='anchors'."
                )


@dataclass(frozen=True)
class RelationAnchorCovarianceArtifact:
    path: Path
    fingerprint: str
    scanned_candidate_count: int
    selected_anchor_count: int
    requested_budget: int
    effective_budget: int
    matched_relation_count: int
    unmatched_relation_count: int
    unallocated_relation_count: int
    allocated_per_relation_min: int
    allocated_per_relation_mean: float
    allocated_per_relation_max: int
    hardness_min: float
    hardness_mean: float
    hardness_max: float


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _uniform_request_selection(
    requests: Sequence[Mapping[str, Any]], max_count: int
) -> List[Mapping[str, Any]]:
    count = len(requests)
    if count <= max_count:
        return list(requests)
    if max_count == 1:
        return [requests[count // 2]]
    indices = [
        (position * (count - 1)) // (max_count - 1)
        for position in range(max_count)
    ]
    return [requests[index] for index in indices]


def _request_fingerprint_payload(
    requests: Sequence[Mapping[str, Any]],
    model_name: str,
    layer_name: str,
    fact_token: str,
    config: AnchorConfig,
) -> Dict[str, Any]:
    return {
        "convention": ANCHOR_COVARIANCE_CONVENTION,
        "model_name": model_name,
        "layer_name": layer_name,
        "requests": [
            {
                "case_id": request.get("case_id"),
                "subject": request["subject"],
                "prompt": request["prompt"],
            }
            for request in requests
        ],
        "anchor_templates": list(config.templates),
        "anchor_max_count": config.max_count,
        "fact_token": fact_token,
        "covariance_dtype": config.covariance_dtype,
    }


def _qa_request_fingerprint_payload(
    selected_requests: Sequence[Mapping[str, Any]],
    model_name: str,
    model_name_or_path: str,
    model_dtype: str,
    layer_name: str,
    fact_token: str,
    config: AnchorConfig,
    candidate_count: int,
    edit_request_count: int,
    scale: float,
) -> Dict[str, Any]:
    return {
        "convention": QA_ANCHOR_COVARIANCE_CONVENTION,
        "context_mode": HELD_OUT_QA_CONTEXT_MODE,
        "model_name": model_name,
        "model_name_or_path": model_name_or_path,
        "model_dtype": model_dtype,
        "layer_name": layer_name,
        "fact_token": fact_token,
        "track": "in",
        "representation_location": "filled_prompt_last_token",
        "covariance_dtype": config.covariance_dtype,
        "selected_requests": [
            {
                "case_id": request["case_id"],
                "prompt": request["prompt"],
                "subject": request["subject"],
            }
            for request in selected_requests
        ],
        "sampling_method": "deterministic_uniform_endpoints_middle_if_one_v1",
        "anchor_max_count": config.max_count,
        "extraction_chunk_size": config.extraction_chunk_size,
        "candidate_count": candidate_count,
        "selected_count": len(selected_requests),
        "effective_mass": edit_request_count,
        "mass_scaling_method": "edit_request_count_over_selected_count",
        "scale": scale,
    }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(
    cache_root: Path, model_name: str, fingerprint: str, layer_name: str
) -> Path:
    safe_layer_name = layer_name.replace("/", "_").replace("\\", "_").replace(".", "_")
    return cache_root / model_name / fingerprint / f"{safe_layer_name}.pt"


def _load_valid_cached_covariance(
    path: Path, fingerprint: str, convention: str = ANCHOR_COVARIANCE_CONVENTION
) -> Optional[torch.Tensor]:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as error:
        print(f"Ignoring unreadable anchor cache {path}: {error}", flush=True)
        return None
    if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
        return None
    if payload.get("convention") != convention:
        return None
    covariance = payload.get("covariance")
    if (
        not isinstance(covariance, torch.Tensor)
        or covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not torch.isfinite(covariance).all().item()
    ):
        return None
    return covariance


def _atomic_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def prepare_anchor_covariances(
    model,
    tok,
    requests: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    module_template: str,
    fact_token: str,
    config: AnchorConfig,
    cache_root: Path,
    neutral_anchor_requests: Optional[Sequence[Mapping[str, Any]]] = None,
    neutral_anchor_context_mode: str = NEUTRAL_TEMPLATE_CONTEXT_MODE,
) -> Dict[str, AnchorCovarianceArtifact]:
    """Precompute every layer artifact before PRCMET starts temporary writes."""

    config.validate(neutral_anchor_context_mode)
    if not requests:
        raise ValueError("Anchor preservation requires at least one edit request.")
    if not fact_token.startswith("subject_"):
        raise ValueError(
            "Anchor preservation requires a subject-based fact_token strategy."
        )
    qa_mode = neutral_anchor_context_mode == HELD_OUT_QA_CONTEXT_MODE
    if qa_mode:
        if not neutral_anchor_requests:
            raise ValueError(
                "zsRE neutral QA anchors require an explicit non-empty held-out "
                "neutral_anchor_requests pool; edit requests cannot be a fallback."
            )
        # The caller excludes the entire experiment. Also reject overlaps with
        # this call's edits for callers supplying their own explicit pool.
        edit_case_ids = {request.get("case_id") for request in requests}
        edit_subjects = {
            " ".join(request["subject"].split()).casefold() for request in requests
        }
        for request in neutral_anchor_requests:
            if request.get("case_id") is None:
                raise ValueError("Every held-out neutral QA anchor requires case_id.")
            for field in ("prompt", "subject"):
                value = request.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Held-out neutral QA anchors require a non-empty {field}."
                    )
            if (
                request["case_id"] in edit_case_ids
                or " ".join(request["subject"].split()).casefold() in edit_subjects
            ):
                raise ValueError(
                    "The held-out neutral QA pool overlaps edit case_ids or "
                    "normalized subjects."
                )
        candidates = neutral_anchor_requests
        convention = QA_ANCHOR_COVARIANCE_CONVENTION
        # An independent namespace as well as a distinct payload convention.
        cache_root = cache_root / HELD_OUT_QA_CONTEXT_MODE
    else:
        candidates = requests
        convention = ANCHOR_COVARIANCE_CONVENTION
    selected_requests = _uniform_request_selection(candidates, config.max_count)
    candidate_count = len(candidates)
    anchor_count = len(selected_requests)
    # Preserve the original effective edit mass and beta interpretation even
    # when the held-out QA candidate pool is much larger than this edit batch.
    scale = len(requests) / anchor_count if qa_mode else candidate_count / anchor_count
    model_name = canonical_stats_model_name(model.config._name_or_path)
    artifacts: Dict[str, AnchorCovarianceArtifact] = {}
    print(f"Neutral anchor context mode: {neutral_anchor_context_mode}", flush=True)
    if qa_mode:
        print(
            "Neutral QA anchor location: last token of the filled held-out question",
            flush=True,
        )
    print(f"Anchor edit requests in this algorithm call: {len(requests)}", flush=True)

    layer_count = len(layers)
    for layer_index, layer in enumerate(layers, start=1):
        layer_name = module_template.format(layer)
        layer_progress = f"[Anchor progress {layer_index}/{layer_count}; actual layer {layer}]"
        print(
            f"{layer_progress} Preparing {layer_name}",
            flush=True,
        )
        if qa_mode:
            fingerprint_payload = _qa_request_fingerprint_payload(
                selected_requests=selected_requests,
                model_name=model_name,
                model_name_or_path=str(model.config._name_or_path),
                model_dtype=str(next(model.parameters()).dtype),
                layer_name=layer_name,
                fact_token=fact_token,
                config=config,
                candidate_count=candidate_count,
                edit_request_count=len(requests),
                scale=scale,
            )
        else:
            # Keep the old template fingerprint and path byte-for-byte compatible.
            fingerprint_payload = _request_fingerprint_payload(
                requests,
                model_name,
                layer_name,
                fact_token,
                config,
            )
        fingerprint = _fingerprint(fingerprint_payload)
        path = _cache_path(cache_root, model_name, fingerprint, layer_name)
        covariance = _load_valid_cached_covariance(path, fingerprint, convention)
        if covariance is None:
            print(
                f"{layer_progress} {layer_name}: Cache miss; "
                f"extracting {anchor_count} anchors",
                flush=True,
            )
            covariance = None
            covariance_dtype = _torch_dtype(config.covariance_dtype)
            with tqdm(
                total=anchor_count,
                desc=f"Anchor {layer_name}",
                unit="anchor",
                dynamic_ncols=True,
            ) as progress:
                for start in range(0, anchor_count, config.extraction_chunk_size):
                    request_chunk = selected_requests[
                        start : start + config.extraction_chunk_size
                    ]
                    if qa_mode:
                        contexts = [request["prompt"] for request in request_chunk]
                        words = [request["subject"] for request in request_chunk]
                    else:
                        contexts = [
                            template
                            for request in request_chunk
                            for template in config.templates
                        ]
                        words = [
                            request["subject"]
                            for request in request_chunk
                            for _ in config.templates
                        ]
                    if qa_mode:
                        # The next answer token is predicted from the final
                        # question token. Protect that task-facing location
                        # rather than the edit-key subject location, which can
                        # directly compete with rewriting while weakly covering
                        # unrelated QA behavior.
                        filled_contexts = [
                            context.format(word)
                            for context, word in zip(contexts, words)
                        ]
                        tokenized = tok(filled_contexts, padding=True)
                        if tok.padding_side == "left":
                            final_token_idxs = [[-1] for _ in filled_contexts]
                        else:
                            final_token_idxs = [
                                [sum(attention_mask) - 1]
                                for attention_mask in tokenized["attention_mask"]
                            ]
                        representations = repr_tools.get_reprs_at_idxs(
                            model=model,
                            tok=tok,
                            contexts=filled_contexts,
                            idxs=final_token_idxs,
                            layer=layer,
                            module_template=module_template,
                            track="in",
                        )
                        anchors = representations
                        del tokenized, filled_contexts, final_token_idxs
                    else:
                        representations = repr_tools.get_reprs_at_word_tokens(
                            model=model,
                            tok=tok,
                            context_templates=contexts,
                            words=words,
                            layer=layer,
                            module_template=module_template,
                            subtoken=fact_token[len("subject_") :],
                            track="in",
                        )
                        anchors = representations.reshape(
                            len(request_chunk), len(config.templates), -1
                        ).mean(dim=1)
                    anchors = anchors.detach().to(dtype=covariance_dtype)
                    if covariance is None:
                        covariance = torch.zeros(
                            (anchors.shape[1], anchors.shape[1]),
                            dtype=covariance_dtype,
                            device=anchors.device,
                        )
                    covariance.addmm_(anchors.T, anchors)
                    del representations, anchors
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    progress.update(len(request_chunk))

            if covariance is None:
                raise RuntimeError("No anchor covariance was produced.")
            covariance.mul_(scale)
            if not torch.isfinite(covariance).all().item():
                raise FloatingPointError("Anchor covariance contains non-finite values.")
            covariance_cpu = covariance.to(device="cpu")
            del covariance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _atomic_save(
                path,
                {
                    "convention": convention,
                    "fingerprint": fingerprint,
                    "metadata": fingerprint_payload,
                    "candidate_count": candidate_count,
                    "anchor_count": anchor_count,
                    "scale": scale,
                    "covariance": covariance_cpu,
                },
            )
            del covariance_cpu
        else:
            del covariance
            print(
                f"{layer_progress} {layer_name}: Cache hit; "
                "skipping extraction",
                flush=True,
            )

        artifacts[layer_name] = AnchorCovarianceArtifact(
            path=path,
            candidate_count=candidate_count,
            anchor_count=anchor_count,
            scale=scale,
            fingerprint=fingerprint,
            convention=convention,
        )
        print(f"Pre-edit anchor cache ready: {path}", flush=True)
        print(f"Anchor candidate count: {candidate_count}", flush=True)
        print(f"Actual anchor count: {anchor_count}", flush=True)
        scaling_rule = "N_edit / N_selected" if qa_mode else "candidate_count / anchor_count"
        print(
            f"Anchor effective mass scale ({scaling_rule}): {scale:.6f}",
            flush=True,
        )

    return artifacts


def load_anchor_covariance(
    artifact: AnchorCovarianceArtifact, reference: torch.Tensor
) -> torch.Tensor:
    covariance = _load_valid_cached_covariance(
        artifact.path, artifact.fingerprint, artifact.convention
    )
    if covariance is None:
        raise RuntimeError(f"Anchor covariance cache is invalid: {artifact.path}")
    if covariance.shape != reference.shape:
        raise ValueError(
            f"Anchor covariance shape {tuple(covariance.shape)} does not match "
            f"the system covariance shape {tuple(reference.shape)}."
        )
    return covariance.to(device=reference.device, dtype=reference.dtype)


def _relation_id(request: Mapping[str, Any], source: str) -> str:
    relation_id = request.get("relation_id")
    if relation_id is None or relation_id == "":
        raise ValueError(
            f"Relation anchor preservation requires relation_id on every {source}."
        )
    return str(relation_id)


def _stable_case_key(request: Mapping[str, Any]) -> Tuple[int, int, str]:
    case_id = request.get("case_id")
    if isinstance(case_id, int) and not isinstance(case_id, bool):
        return 0, case_id, ""
    case_text = str(case_id)
    if case_text.isdigit():
        return 0, int(case_text), case_text
    return 1, 0, case_text


def _stable_relation_key(relation_id: str) -> str:
    return str(relation_id)


def allocate_relation_anchor_budget(
    edit_counts: Mapping[str, int],
    candidate_counts: Mapping[str, int],
    total_budget: int,
) -> Dict[str, int]:
    """Deterministically apportion a global budget with candidate capacities."""

    if (
        isinstance(total_budget, bool)
        or not isinstance(total_budget, int)
        or total_budget <= 0
    ):
        raise ValueError("relation_anchor_budget must be a positive integer.")
    for mapping_name, counts in (
        ("edit_counts", edit_counts),
        ("candidate_counts", candidate_counts),
    ):
        for relation, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    f"{mapping_name}[{relation!r}] must be a non-negative integer."
                )
    relations = sorted(edit_counts, key=_stable_relation_key)
    allocations = {relation: 0 for relation in relations}
    valid_relations = [
        relation
        for relation in relations
        if edit_counts[relation] > 0 and candidate_counts.get(relation, 0) > 0
    ]
    if not valid_relations:
        return allocations

    capacity = sum(candidate_counts[relation] for relation in valid_relations)
    effective_budget = min(total_budget, capacity)
    if effective_budget < len(valid_relations):
        prioritized = sorted(
            valid_relations,
            key=lambda relation: (
                -edit_counts[relation],
                _stable_relation_key(relation),
            ),
        )
        for relation in prioritized[:effective_budget]:
            allocations[relation] = 1
        return allocations

    total_edits = sum(edit_counts.values())
    if total_edits <= 0:
        raise ValueError("Relation anchor allocation requires positive edit counts.")
    quotas = {
        relation: total_budget * edit_counts[relation] / total_edits
        for relation in valid_relations
    }
    for relation in valid_relations:
        allocations[relation] = min(
            candidate_counts[relation], math.floor(quotas[relation])
        )

    # A usable relation receives at least one anchor whenever the budget permits.
    for relation in valid_relations:
        if allocations[relation] == 0:
            allocations[relation] = 1

    # Minimum-one constraints can overfill a Hamilton floor allocation. Remove
    # the greatest quota excess first while never taking a relation below one.
    while sum(allocations.values()) > effective_budget:
        donors = [
            relation
            for relation in valid_relations
            if allocations[relation] > 1
        ]
        if not donors:
            raise RuntimeError("Unable to satisfy relation anchor budget.")
        donor = sorted(
            donors,
            key=lambda relation: (
                -(allocations[relation] - quotas[relation]),
                _stable_relation_key(relation),
            ),
        )[0]
        allocations[donor] -= 1

    # Largest-remainder completion of the ideal edit-frequency quotas.
    remainder_order = sorted(
        valid_relations,
        key=lambda relation: (
            -(quotas[relation] - math.floor(quotas[relation])),
            _stable_relation_key(relation),
        ),
    )
    for relation in remainder_order:
        if sum(allocations.values()) >= effective_budget:
            break
        if allocations[relation] < candidate_counts[relation]:
            allocations[relation] += 1

    # Redistribute capacity left unused by saturated or unmatched relations.
    while sum(allocations.values()) < effective_budget:
        recipients = [
            relation
            for relation in valid_relations
            if allocations[relation] < candidate_counts[relation]
        ]
        if not recipients:
            break
        recipient = sorted(
            recipients,
            key=lambda relation: (
                allocations[relation] / edit_counts[relation],
                _stable_relation_key(relation),
            ),
        )[0]
        allocations[recipient] += 1

    allocated_total = sum(allocations.values())
    if allocated_total != effective_budget or any(
        count < 0 or count > candidate_counts.get(relation, 0)
        for relation, count in allocations.items()
    ):
        raise RuntimeError("Invalid relation anchor budget allocation.")
    return allocations


def _relation_fingerprint_payload(
    edit_requests: Sequence[Mapping[str, Any]],
    preservation_requests: Sequence[Mapping[str, Any]],
    model_name: str,
    layer_name: str,
    fact_token: str,
    config: RelationAnchorConfig,
) -> Dict[str, Any]:
    def projection(request: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "case_id": request.get("case_id"),
            "relation_id": request.get("relation_id"),
            "subject": request["subject"],
            "prompt": request["prompt"],
        }

    return {
        "convention": RELATION_ANCHOR_COVARIANCE_CONVENTION,
        "hardness_method": RELATION_ANCHOR_HARDNESS_METHOD,
        "model_name": model_name,
        "layer_name": layer_name,
        "fact_token": fact_token,
        "covariance_dtype": config.covariance_dtype,
        "edit_requests": [projection(request) for request in edit_requests],
        "scanned_anchor_requests": [
            projection(request) for request in preservation_requests
        ],
        "relation_anchor_budget": config.budget,
        "anchor_max_count": config.max_count,
        "budget_allocation_method": RELATION_ANCHOR_BUDGET_METHOD,
    }


def _load_valid_relation_payload(
    path: Path, fingerprint: str
) -> Optional[Mapping[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as error:
        print(f"Ignoring unreadable relation anchor cache {path}: {error}", flush=True)
        return None
    if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
        return None
    if payload.get("convention") != RELATION_ANCHOR_COVARIANCE_CONVENTION:
        return None
    covariance = payload.get("covariance")
    if (
        not isinstance(covariance, torch.Tensor)
        or covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not torch.isfinite(covariance).all().item()
    ):
        return None
    return payload


def _relation_artifact_from_payload(
    path: Path, fingerprint: str, payload: Mapping[str, Any]
) -> RelationAnchorCovarianceArtifact:
    stats = payload.get("hardness_stats", {})
    allocated_stats = payload.get("allocated_per_relation_stats", {})
    return RelationAnchorCovarianceArtifact(
        path=path,
        fingerprint=fingerprint,
        scanned_candidate_count=int(payload["scanned_candidate_count"]),
        selected_anchor_count=int(payload["selected_anchor_count"]),
        requested_budget=int(payload["requested_budget"]),
        effective_budget=int(payload["effective_budget"]),
        matched_relation_count=int(payload["matched_relation_count"]),
        unmatched_relation_count=int(payload["unmatched_relation_count"]),
        unallocated_relation_count=int(payload["unallocated_relation_count"]),
        allocated_per_relation_min=int(allocated_stats.get("min", 0)),
        allocated_per_relation_mean=float(allocated_stats.get("mean", 0.0)),
        allocated_per_relation_max=int(allocated_stats.get("max", 0)),
        hardness_min=float(stats.get("min", 0.0)),
        hardness_mean=float(stats.get("mean", 0.0)),
        hardness_max=float(stats.get("max", 0.0)),
    )


def _safe_unit_rows(values: torch.Tensor, eps: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    return torch.where(
        norms > eps,
        values / norms.clamp_min(eps),
        torch.zeros_like(values),
    )


def _finite_rows_or_zero(values: torch.Tensor) -> torch.Tensor:
    finite_rows = torch.isfinite(values).all(dim=1, keepdim=True)
    return torch.where(finite_rows, values, torch.zeros_like(values))


def prepare_relation_anchor_covariances(
    model,
    tok,
    edit_requests: Sequence[Mapping[str, Any]],
    preservation_anchor_requests: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    module_template: str,
    fact_token: str,
    config: RelationAnchorConfig,
    cache_root: Path,
) -> Dict[str, RelationAnchorCovarianceArtifact]:
    """Build held-out, same-relation hard-anchor covariances pre-edit."""

    config.validate(preservation_constraint=True, preservation_mode="anchors")
    if not config.enabled:
        return {}
    if not edit_requests:
        raise ValueError("Relation anchor preservation requires edit requests.")
    if not preservation_anchor_requests:
        raise ValueError(
            "Relation anchor preservation requires a non-empty held-out anchor pool."
        )
    if not fact_token.startswith("subject_"):
        raise ValueError(
            "Relation anchor preservation requires a subject-based fact_token strategy."
        )

    edit_ids = {request.get("case_id") for request in edit_requests}
    if None in edit_ids:
        raise ValueError("Every edit request must have a case_id.")
    for request in edit_requests:
        _relation_id(request, "edit request")
        if "subject" not in request or "prompt" not in request:
            raise ValueError("Edit requests require subject and prompt fields.")
    for request in preservation_anchor_requests:
        _relation_id(request, "held-out anchor request")
        if request.get("case_id") is None:
            raise ValueError("Every held-out anchor request must have a case_id.")
        if request.get("case_id") in edit_ids:
            raise ValueError(
                "The relation anchor pool overlaps the edit case_id set."
            )
        if "subject" not in request or "prompt" not in request:
            raise ValueError(
                "Held-out anchor requests require subject and prompt fields."
            )

    edit_relation_ids = {
        _relation_id(request, "edit request") for request in edit_requests
    }
    eligible_pool = sorted(
        [
            request
            for request in preservation_anchor_requests
            if _relation_id(request, "held-out anchor request")
            in edit_relation_ids
        ],
        key=_stable_case_key,
    )
    if not eligible_pool:
        raise ValueError(
            "No held-out facts match any relation_id in the edit requests."
        )
    sampled_pool = _uniform_request_selection(eligible_pool, config.max_count)
    scanned_counts_by_relation: Dict[str, int] = {
        relation: 0 for relation in edit_relation_ids
    }
    for request in sampled_pool:
        relation = _relation_id(request, "held-out anchor request")
        scanned_counts_by_relation[relation] += 1
    model_name = canonical_stats_model_name(model.config._name_or_path)
    covariance_dtype = _torch_dtype(config.covariance_dtype)
    artifacts: Dict[str, RelationAnchorCovarianceArtifact] = {}

    for layer_index, layer in enumerate(layers, start=1):
        layer_name = module_template.format(layer)
        print(
            f"[Relation anchor layer {layer_index}/{len(layers)}] {layer_name}",
            flush=True,
        )
        fingerprint_payload = _relation_fingerprint_payload(
            edit_requests,
            sampled_pool,
            model_name,
            layer_name,
            fact_token,
            config,
        )
        fingerprint = _fingerprint(fingerprint_payload)
        path = _cache_path(cache_root, model_name, fingerprint, layer_name)
        cached_payload = _load_valid_relation_payload(path, fingerprint)
        if cached_payload is not None:
            artifact = _relation_artifact_from_payload(
                path, fingerprint, cached_payload
            )
            artifacts[layer_name] = artifact
            print(
                f"[Relation anchor layer {layer_index}/{len(layers)}] Cache hit",
                flush=True,
            )
            print(
                f"Relation candidates scanned: {artifact.scanned_candidate_count}; "
                f"selected: {artifact.selected_anchor_count}",
                flush=True,
            )
            del cached_payload
            continue

        print(
            f"[Relation anchor layer {layer_index}/{len(layers)}] Cache miss",
            flush=True,
        )
        relation_sums: Dict[str, torch.Tensor] = {}
        edit_counts: Dict[str, int] = {}
        representation_device = None
        with tqdm(
            total=len(edit_requests),
            desc=f"Relation centers {layer_name}",
            unit="edit",
            dynamic_ncols=True,
        ) as progress:
            for start in range(0, len(edit_requests), config.extraction_chunk_size):
                chunk = edit_requests[start : start + config.extraction_chunk_size]
                representations = repr_tools.get_reprs_at_word_tokens(
                    model=model,
                    tok=tok,
                    context_templates=[request["prompt"] for request in chunk],
                    words=[request["subject"] for request in chunk],
                    layer=layer,
                    module_template=module_template,
                    subtoken=fact_token[len("subject_") :],
                    track="in",
                ).reshape(len(chunk), -1)
                representation_device = representations.device
                keys = _finite_rows_or_zero(
                    representations.detach().to(dtype=covariance_dtype)
                )
                eps = torch.finfo(keys.dtype).eps
                normalized = _safe_unit_rows(keys, eps)
                for row, request in zip(normalized, chunk):
                    relation = _relation_id(request, "edit request")
                    if relation not in relation_sums:
                        relation_sums[relation] = torch.zeros_like(row)
                        edit_counts[relation] = 0
                    relation_sums[relation].add_(row)
                    edit_counts[relation] += 1
                del representations, keys, normalized
                progress.update(len(chunk))
                progress.set_postfix(relations=len(relation_sums))

        centers: Dict[str, torch.Tensor] = {}
        center_eps = torch.finfo(covariance_dtype).eps
        invalid_center_count = 0
        for relation, relation_sum in relation_sums.items():
            norm = torch.linalg.vector_norm(relation_sum)
            if torch.isfinite(norm).item() and norm.item() > center_eps:
                centers[relation] = relation_sum / norm
            else:
                invalid_center_count += 1
        del relation_sums
        if invalid_center_count:
            print(
                f"Relations with unusable edit centers: {invalid_center_count}",
                flush=True,
            )
        if not centers:
            raise ValueError("All edit relation centers are zero or non-finite.")

        usable_candidate_counts = {
            relation: (
                scanned_counts_by_relation.get(relation, 0)
                if relation in centers
                else 0
            )
            for relation in edit_counts
        }
        allocations = allocate_relation_anchor_budget(
            edit_counts, usable_candidate_counts, config.budget
        )
        unmatched_relations = [
            relation
            for relation in edit_counts
            if usable_candidate_counts[relation] == 0
        ]
        unallocated_relations = [
            relation
            for relation in edit_counts
            if usable_candidate_counts[relation] > 0
            and allocations[relation] == 0
        ]
        if unmatched_relations:
            print(
                f"Relations without usable candidates: {len(unmatched_relations)}",
                flush=True,
            )
        allocated_total = sum(allocations.values())
        if allocated_total == 0:
            raise ValueError(
                "No relation can receive an anchor under the configured budget."
            )
        if unallocated_relations:
            print(
                f"Relation anchor budget left {len(unallocated_relations)} "
                "candidate-bearing relations without an anchor.",
                flush=True,
            )

        candidate_scan_pool = [
            request
            for request in sampled_pool
            if allocations[_relation_id(request, "held-out anchor request")] > 0
        ]
        top_by_relation: Dict[
            str, List[Tuple[float, Tuple[int, int, str], torch.Tensor]]
        ] = {
            relation: []
            for relation, allocation in allocations.items()
            if allocation > 0
        }
        with tqdm(
            total=len(candidate_scan_pool),
            desc=f"Relation candidates {layer_name}",
            unit="candidate",
            dynamic_ncols=True,
        ) as progress:
            for start in range(
                0, len(candidate_scan_pool), config.extraction_chunk_size
            ):
                chunk = candidate_scan_pool[
                    start : start + config.extraction_chunk_size
                ]
                representations = repr_tools.get_reprs_at_word_tokens(
                    model=model,
                    tok=tok,
                    context_templates=[request["prompt"] for request in chunk],
                    words=[request["subject"] for request in chunk],
                    layer=layer,
                    module_template=module_template,
                    subtoken=fact_token[len("subject_") :],
                    track="in",
                ).reshape(len(chunk), -1)
                representation_device = representations.device
                keys = _finite_rows_or_zero(
                    representations.detach().to(dtype=covariance_dtype)
                )
                normalized = _safe_unit_rows(keys, torch.finfo(keys.dtype).eps)
                for row_index, request in enumerate(chunk):
                    relation = _relation_id(request, "held-out anchor request")
                    center = centers.get(relation)
                    if center is None:
                        continue
                    score_tensor = torch.abs(torch.dot(normalized[row_index], center))
                    score = (
                        float(score_tensor.item())
                        if torch.isfinite(score_tensor).item()
                        else 0.0
                    )
                    entries = top_by_relation[relation]
                    entries.append(
                        (score, _stable_case_key(request), keys[row_index].cpu())
                    )
                    entries.sort(key=lambda item: (-item[0], item[1]))
                    del entries[allocations[relation] :]
                del representations, keys, normalized
                progress.update(len(chunk))
                progress.set_postfix(
                    candidates=progress.n,
                    kept=sum(len(entries) for entries in top_by_relation.values()),
                )

        selected_counts = {
            relation: len(top_by_relation.get(relation, []))
            for relation in edit_counts
        }
        selected_count = sum(selected_counts.values())
        if selected_count == 0:
            raise ValueError(
                "No relation anchor was selected for any edit relation."
            )
        if selected_count != allocated_total:
            raise RuntimeError(
                "Relation anchor extraction did not satisfy its deterministic allocation."
            )

        if representation_device is None:
            raise RuntimeError("Relation anchor extraction produced no representations.")
        feature_size = next(
            entry[2].numel()
            for entries in top_by_relation.values()
            for entry in entries
        )
        covariance = torch.zeros(
            (feature_size, feature_size),
            dtype=covariance_dtype,
            device=representation_device,
        )
        selected_scores: List[float] = []
        with tqdm(
            total=selected_count,
            desc=f"Relation covariance {layer_name}",
            unit="anchor",
            dynamic_ncols=True,
        ) as progress:
            for relation in sorted(top_by_relation):
                entries = top_by_relation[relation]
                if not entries:
                    continue
                relation_scale = edit_counts[relation] / len(entries)
                for start in range(0, len(entries), config.extraction_chunk_size):
                    entry_chunk = entries[
                        start : start + config.extraction_chunk_size
                    ]
                    scores = [entry[0] for entry in entry_chunk]
                    anchors = torch.stack(
                        [entry[2] for entry in entry_chunk]
                    ).to(device=representation_device, dtype=covariance_dtype)
                    scaled_anchors = anchors * math.sqrt(relation_scale)
                    covariance.addmm_(scaled_anchors.T, scaled_anchors)
                    selected_scores.extend(scores)
                    progress.update(len(entry_chunk))
                    progress.set_postfix(selected=len(selected_scores))
                    del anchors, scaled_anchors

        if not torch.isfinite(covariance).all().item():
            raise FloatingPointError(
                "Relation anchor covariance contains non-finite values."
            )
        matched_allocated_counts = [
            count for count in selected_counts.values() if count > 0
        ]
        allocated_stats = {
            "min": min(matched_allocated_counts),
            "mean": sum(matched_allocated_counts) / len(matched_allocated_counts),
            "max": max(matched_allocated_counts),
        }
        hardness_stats = {
            "min": min(selected_scores),
            "mean": sum(selected_scores) / len(selected_scores),
            "max": max(selected_scores),
        }
        covariance_cpu = covariance.cpu()
        del covariance, centers, top_by_relation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        payload = {
            "convention": RELATION_ANCHOR_COVARIANCE_CONVENTION,
            "fingerprint": fingerprint,
            "metadata": fingerprint_payload,
            "scanned_candidate_count": len(candidate_scan_pool),
            "selected_anchor_count": selected_count,
            "requested_budget": config.budget,
            "effective_budget": selected_count,
            "matched_relation_count": len(matched_allocated_counts),
            "unmatched_relation_count": len(unmatched_relations),
            "unallocated_relation_count": len(unallocated_relations),
            "edit_counts_by_relation": edit_counts,
            "candidate_counts_by_relation": scanned_counts_by_relation,
            "allocated_counts_by_relation": selected_counts,
            "allocated_per_relation_stats": allocated_stats,
            "hardness_stats": hardness_stats,
            "covariance": covariance_cpu,
        }
        _atomic_save(path, payload)
        artifact = _relation_artifact_from_payload(path, fingerprint, payload)
        artifacts[layer_name] = artifact
        print(
            f"Relation candidates scanned: {len(candidate_scan_pool)}; "
            f"selected: {selected_count}/{config.budget}",
            flush=True,
        )
        print(f"Relation anchor cache ready: {path}", flush=True)
        del payload, covariance_cpu

    return artifacts


def load_relation_anchor_covariance(
    artifact: RelationAnchorCovarianceArtifact, reference: torch.Tensor
) -> torch.Tensor:
    payload = _load_valid_relation_payload(artifact.path, artifact.fingerprint)
    if payload is None:
        raise RuntimeError(
            f"Relation anchor covariance cache is invalid: {artifact.path}"
        )
    covariance = payload["covariance"]
    if covariance.shape != reference.shape:
        raise ValueError(
            f"Relation anchor covariance shape {tuple(covariance.shape)} does not "
            f"match system covariance shape {tuple(reference.shape)}."
        )
    return covariance.to(device=reference.device, dtype=reference.dtype)
