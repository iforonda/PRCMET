# new methods experience
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *
from util.model_name import canonical_stats_model_name

from .anchors import (
    AnchorConfig,
    AnchorCovarianceArtifact,
    NEUTRAL_TEMPLATE_CONTEXT_MODE,
    RelationAnchorConfig,
    RelationAnchorCovarianceArtifact,
    load_anchor_covariance,
    load_relation_anchor_covariance,
    prepare_anchor_covariances,
    prepare_relation_anchor_covariances,
)
from .compute_ks import compute_ks, compute_ks_parallel
from .compute_zs import compute_zs, compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .prcmet_hparams import PRCMETHyperParams
from .preservation import (
    PendingPreservationUpdate,
    PreservationConfig,
    PreservationState,
    ResponseCalibrationConfig,
    ResponseCalibrationDiagnostics,
    bounded_response_calibrated_update,
    factorize_right_system,
    right_solve_update,
)

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
KZ_CACHE= {}
PRESERVATION_STATE = PreservationState()
# Backward-compatible view for callers that inspected the old cache directly.
PRESERVE_COV_CACHE = PRESERVATION_STATE.covariances


def reset_preservation_state() -> None:
    """Start an independent PRCMET preservation run."""

    PRESERVATION_STATE.reset()


def get_preservation_state_dict() -> Dict[str, Any]:
    """Return a CPU checkpoint containing fixed-beta covariance state."""

    return PRESERVATION_STATE.state_dict()


def load_preservation_state_dict(
    state: Dict[str, Any], expected_model_name: Optional[str] = None
) -> None:
    """Restore a compatible preservation checkpoint."""

    normalized_model_name = (
        canonical_stats_model_name(expected_model_name)
        if expected_model_name is not None
        else None
    )
    PRESERVATION_STATE.load_state_dict(state, normalized_model_name)


def apply_prcmet_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PRCMETHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    preservation_anchor_requests: Optional[List[Dict]] = None,
    neutral_anchor_requests: Optional[Sequence[Mapping[str, Any]]] = None,
    neutral_anchor_context_mode: str = NEUTRAL_TEMPLATE_CONTEXT_MODE,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    
    deltas, pending_preservation_updates = execute_prcmet(
        model,
        tok,
        requests,
        hparams,
        cache_template=cache_template,
        preservation_anchor_requests=preservation_anchor_requests,
        neutral_anchor_requests=neutral_anchor_requests,
        neutral_anchor_context_mode=neutral_anchor_context_mode,
        return_preservation_updates=True,
    )  # stores the parameter updates around Eq. 14

    rollback_weights = {}
    try:
        with torch.no_grad():
            for w_name, upd_matrix in deltas.items():  # w_name, update
                upd_matrix = upd_matrix.to("cuda")
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                original = w.detach().clone()
                rollback_weights[w_name] = original
                if return_orig_weights:
                    weights_copy[w_name] = original

                w[...] += upd_matrix.float()

        # History mode advances only after all requested parameter writes
        # succeed. Anchor mode never reads or mutates this cross-batch state.
        preservation_config = PreservationConfig.from_hparams(hparams)
        if preservation_config.enabled and preservation_config.mode == "history":
            PRESERVATION_STATE.commit(pending_preservation_updates)
        elif pending_preservation_updates:
            raise RuntimeError(
                "Non-history preservation unexpectedly staged history updates."
            )
    except Exception:
        with torch.no_grad():
            for w_name, original in rollback_weights.items():
                nethook.get_parameter(model, w_name)[...] = original
        raise

    print(f"\nNew weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_prcmet(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PRCMETHyperParams,
    cache_template: Optional[str] = None,
    preservation_anchor_requests: Optional[List[Dict]] = None,
    return_preservation_updates: bool = False,
    neutral_anchor_requests: Optional[Sequence[Mapping[str, Any]]] = None,
    neutral_anchor_context_mode: str = NEUTRAL_TEMPLATE_CONTEXT_MODE,
) -> Union[
    Dict[str, torch.Tensor],
    Tuple[Dict[str, torch.Tensor], List[PendingPreservationUpdate]],
]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}
    pending_preservation_updates: List[PendingPreservationUpdate] = []
    # Update target and print info
    requests = deepcopy(requests)
    print(f"Total edit requests: {len(requests)}", flush=True)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT_ATTN request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter( # transformer.h.{}.attn.out_proj
            model, f"{rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
        for rewrite_module_tmp in hparams.rewrite_module_tmps
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    rewrite_module_names = hparams.rewrite_module_tmps

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = dict()
    for rewrite_module_name in rewrite_module_names:
        z_list[rewrite_module_name] = []
    # get zs
    for request in requests:
        # Retrieve k/v pair if already stored in cache
        for rewrite_module_name in rewrite_module_names:
            block_name = "attn" if "attn" in rewrite_module_name else "mlp"
            cache_fname = (
                Path(
                    str(cache_template).format(
                        z_layer, block_name, hparams.clamp_norm_factor, request["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False
            if (
                cache_fname is not None  # Require cache template
                and cache_fname.exists()  # Cache file must exist
            ):
                try:
                    data = np.load(cache_fname)
                    z_list[rewrite_module_name].append(torch.from_numpy(data["v_star"]).to("cuda"))
                    data_loaded = True
                except Exception as e:
                    print(f"Error reading cache file due to {e}. Recomputing...")

            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                if len(rewrite_module_names) == 2:
                    cur_z_attn, cur_z_mlp = compute_zs( 
                            model,
                            tok,
                            request,
                            hparams,
                            z_layer,
                            context_templates,
                    )
                    z_list[rewrite_module_names[0]].append(cur_z_attn if "attn" in rewrite_module_names[0] else cur_z_mlp)
                    z_list[rewrite_module_names[1]].append(cur_z_attn if "attn" in rewrite_module_names[1] else cur_z_mlp)
                    for rewrite_module_name in rewrite_module_names:
                        block_name = "attn" if "attn" in rewrite_module_name else "mlp"
                        cache_fname = (
                            Path(
                                str(cache_template).format(
                                    z_layer, block_name, hparams.clamp_norm_factor, request["case_id"]
                                )
                            )
                            if cache_template is not None
                            else None
                        )
                        if cache_fname is not None:
                            cache_fname.parent.mkdir(exist_ok=True, parents=True)
                            if block_name == "attn":
                                np.savez(
                                    cache_fname,
                                    **{
                                        "v_star": cur_z_attn.detach().cpu().numpy(),
                                    },
                                )
                            else:
                                np.savez(
                                    cache_fname,
                                    **{
                                        "v_star": cur_z_mlp.detach().cpu().numpy(),
                                    },
                                )
                            print(f"Cached k/v pair at {cache_fname}")
                else:
                    cur_z_attn, cur_z_mlp = compute_zs( 
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                )
                    if "attn" == block_name:
                        cur_z = cur_z_attn
                    else:
                        cur_z = cur_z_mlp
                    z_list[rewrite_module_name].append(cur_z)
                    if cache_fname is not None:
                        cache_fname.parent.mkdir(exist_ok=True, parents=True)
                        np.savez(
                            cache_fname,
                            **{
                                "v_star": cur_z.detach().cpu().numpy(),
                            },
                        )
                        print(f"Cached k/v pair at {cache_fname}")
                break

    for k, v in z_list.items():
        z_list[k] = torch.stack(v, dim=1)

    preservation_config = PreservationConfig.from_hparams(hparams)
    preservation_enabled = preservation_config.enabled
    if preservation_enabled:
        print("Preservation constraint enabled", flush=True)
        print(f"Preservation mode: {preservation_config.mode}", flush=True)
        print(f"beta_preserve: {preservation_config.beta}", flush=True)
        if preservation_config.mode == "history":
            print(f"preserve_rho: {preservation_config.rho}", flush=True)

    response_calibration_config = ResponseCalibrationConfig.from_hparams(hparams)
    response_calibration_enabled = response_calibration_config.enabled
    if response_calibration_enabled:
        print("Bounded response calibration enabled", flush=True)
        print(
            f"response_calibration_alpha: {response_calibration_config.alpha}",
            flush=True,
        )
        print(
            "response_calibration_max_ratio: "
            f"{response_calibration_config.max_ratio}",
            flush=True,
        )
        print(
            "response calibration solve chunk size: "
            f"{response_calibration_config.solve_chunk_size}",
            flush=True,
        )

    relation_anchor_config = RelationAnchorConfig.from_hparams(hparams)
    relation_anchor_artifacts: Dict[
        str, RelationAnchorCovarianceArtifact
    ] = {}
    if relation_anchor_config.enabled:
        if hparams.mlp_module_tmp not in rewrite_module_names:
            raise ValueError(
                "Relation anchor preservation requires mlp_module_tmp to be an "
                "active rewrite module."
            )
        if preservation_anchor_requests is None:
            raise ValueError(
                "Relation anchor preservation requires an explicit held-out "
                "preservation_anchor_requests pool."
            )
        print("Relation-conditioned hard anchor preservation enabled", flush=True)
        print(
            f"relation_anchor_beta: {relation_anchor_config.beta}", flush=True
        )
        print(
            f"relation_anchor_budget: {relation_anchor_config.budget}",
            flush=True,
        )

    anchor_artifacts: Dict[str, AnchorCovarianceArtifact] = {}
    if preservation_enabled and preservation_config.mode == "anchors":
        if hparams.mlp_module_tmp not in rewrite_module_names:
            raise ValueError(
                "Anchor preservation requires mlp_module_tmp to be an active "
                "rewrite module."
            )
        anchor_config = AnchorConfig.from_hparams(hparams, neutral_anchor_context_mode)
        # This completes for every layer before the Insert loop can make any
        # temporary PRCMET weight change.
        anchor_artifacts = prepare_anchor_covariances(
            model=model,
            tok=tok,
            requests=requests,
            layers=hparams.layers,
            module_template=hparams.mlp_module_tmp,
            fact_token=hparams.fact_token,
            config=anchor_config,
            cache_root=STATS_DIR / "anchor_stats",
            neutral_anchor_requests=neutral_anchor_requests,
            neutral_anchor_context_mode=neutral_anchor_context_mode,
        )

    if relation_anchor_config.enabled:
        # Like neutral anchors, all layers are cached while model weights still
        # equal the original pre-edit state.
        relation_anchor_artifacts = prepare_relation_anchor_covariances(
            model=model,
            tok=tok,
            edit_requests=requests,
            preservation_anchor_requests=preservation_anchor_requests,
            layers=hparams.layers,
            module_template=hparams.mlp_module_tmp,
            fact_token=hparams.fact_token,
            config=relation_anchor_config,
            cache_root=STATS_DIR / "relation_anchor_stats",
        )

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n") 
        layers_ks = None
        # force_recompute = layer != hparams.layers[0]
        for rewrite_module_name in rewrite_module_names:
            # Get current model activations
            history_preserve_this_update = (
                preservation_enabled
                and preservation_config.mode == "history"
                and rewrite_module_name == hparams.mlp_module_tmp
            )
            anchor_preserve_this_update = (
                preservation_enabled
                and preservation_config.mode == "anchors"
                and rewrite_module_name == hparams.mlp_module_tmp
            )
            preserve_this_update = (
                history_preserve_this_update or anchor_preserve_this_update
            )
            relation_preserve_this_update = (
                relation_anchor_config.enabled
                and rewrite_module_name == hparams.mlp_module_tmp
            )
            calibrate_this_update = (
                response_calibration_enabled
                and rewrite_module_name == hparams.mlp_module_tmp
            )

            if 'gpt-j' in model.config._name_or_path and len(rewrite_module_names) == 2:
                if layers_ks == None:
                    layers_ks = compute_ks_parallel(model, tok, requests, hparams, layer, context_templates)  #K eqn 19
            else:
                layers_ks = compute_ks(model, tok, requests, hparams, rewrite_module_name, layer, context_templates)

            print(f"Writing {layers_ks[rewrite_module_name].size(0)} key/value pair(s) into layers")
            cur_zs = get_module_input_output_at_words( # hidden states eqn 2
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=rewrite_module_name,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = z_list[rewrite_module_name]  - cur_zs #z_i - h_i^L
            try:
                layer_ks, targets = (
                    layers_ks[rewrite_module_name].T.double().to("cuda:1"),
                    targets.double().to("cuda:1")
                )
            except:
                layer_ks, targets = (
                    layers_ks[rewrite_module_name].T.double(),
                    targets.double()
                )
            # Load covariance matrix
            force_recompute = False
            # force_recompute = layer != hparams.layers[0]
            cov = get_cov(
                model,
                tok,
                rewrite_module_name.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples
                if not force_recompute
                else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype,
                force_recompute=force_recompute,
            )

            repeat_factor = (layer_ks.size(1) // targets.size(1))
            targets = targets.repeat_interleave(repeat_factor, dim=1) #r
            preserve_cov_norm = None
            preservation_penalty = None
            neutral_cov_norm = None
            neutral_preservation_penalty = None
            relation_cov_norm = None
            relation_preservation_penalty = None
            base_system_norm = None
            neutral_ratio = None
            relation_ratio = None
            relation_artifact = None
            calibration_diagnostics: Optional[ResponseCalibrationDiagnostics] = None
            if (
                preserve_this_update
                or relation_preserve_this_update
                or calibrate_this_update
            ):
                key_cov = layer_ks @ layer_ks.T
                system_matrix = (
                    key_cov + hparams.mom2_update_weight * cov.double()
                )
                base_system_norm = torch.linalg.norm(system_matrix).item()
                if history_preserve_this_update:
                    # Stage this Gram now, but expose it to future batches only
                    # after every requested model weight has been written.
                    layer_name = rewrite_module_name.format(layer)
                    model_name = canonical_stats_model_name(model.config._name_or_path)
                    preserve_cov_device, pending_update = (
                        PRESERVATION_STATE.stage(
                            (model_name, layer_name),
                            key_cov,
                            cov,
                            preservation_config,
                        )
                    )
                    pending_preservation_updates.append(pending_update)
                    preserve_cov_norm = torch.linalg.norm(preserve_cov_device).item()
                    preservation_penalty = (
                        preservation_config.beta * preserve_cov_norm
                    )
                    preserve_cov_device = preserve_cov_device.to(
                        device=key_cov.device, dtype=key_cov.dtype
                    )
                    system_matrix.add_(
                        preserve_cov_device, alpha=preservation_config.beta
                    )
                    del preserve_cov_device
                elif anchor_preserve_this_update:
                    layer_name = rewrite_module_name.format(layer)
                    artifact = anchor_artifacts[layer_name]
                    anchor_cov_device = load_anchor_covariance(artifact, key_cov)
                    preserve_cov_norm = torch.linalg.norm(anchor_cov_device).item()
                    neutral_cov_norm = preserve_cov_norm
                    print(
                        f"Anchor covariance norm: {preserve_cov_norm:.6f}",
                        flush=True,
                    )
                    preservation_penalty = (
                        preservation_config.beta * preserve_cov_norm
                    )
                    neutral_preservation_penalty = preservation_penalty
                    neutral_ratio = preservation_penalty / max(
                        base_system_norm, torch.finfo(system_matrix.dtype).eps
                    )
                    system_matrix.add_(
                        anchor_cov_device, alpha=preservation_config.beta
                    )
                    del anchor_cov_device

                if relation_preserve_this_update:
                    layer_name = rewrite_module_name.format(layer)
                    relation_artifact = relation_anchor_artifacts[layer_name]
                    relation_cov_device = load_relation_anchor_covariance(
                        relation_artifact, key_cov
                    )
                    relation_cov_norm = torch.linalg.norm(
                        relation_cov_device
                    ).item()
                    relation_preservation_penalty = (
                        relation_anchor_config.beta * relation_cov_norm
                    )
                    relation_ratio = relation_preservation_penalty / max(
                        base_system_norm, torch.finfo(system_matrix.dtype).eps
                    )
                    system_matrix.add_(
                        relation_cov_device, alpha=relation_anchor_config.beta
                    )
                    del relation_cov_device

                del key_cov, cov
                scaled_targets = targets / np.sqrt(len(hparams.layers) - i)
                if calibrate_this_update:
                    factorization = factorize_right_system(system_matrix)
                    del system_matrix
                    upd_matrix, calibration_diagnostics = (
                        bounded_response_calibrated_update(
                            scaled_targets,
                            layer_ks,
                            factorization,
                            response_calibration_config,
                        )
                    )
                    del factorization
                else:
                    # Preserve the existing fixed-beta solve when calibration is off.
                    upd_matrix = right_solve_update(
                        scaled_targets,
                        layer_ks,
                        system_matrix,
                    )
                    del system_matrix
                del scaled_targets
            else:
                # Exact original PRCMET square-root residual spreading and update.
                upd_matrix =  (targets / np.sqrt((len(hparams.layers) - i ))) @ layer_ks.T @ torch.inverse(layer_ks @ layer_ks.T + 
                                                 hparams.mom2_update_weight * cov.double())
            weight_name = f"{rewrite_module_name.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            if (
                preserve_this_update
                or relation_preserve_this_update
                or calibrate_this_update
            ) and not torch.isfinite(upd_matrix).all().item():
                raise FloatingPointError(
                    f"Non-finite PRCMET update for {weight_name}; staged preservation "
                    "state, if any, was not committed."
                )

            print(weight_name, ":\norig norm", torch.linalg.norm(weights[weight_name]))
            print("upd norm", torch.linalg.norm(upd_matrix))
            if preserve_this_update:
                print(f"Layer: {layer}", flush=True)
                print(
                    f"update norm: {torch.linalg.norm(upd_matrix).item():.6f}",
                    flush=True,
                )
                print(
                    f"preserve covariance norm: {preserve_cov_norm:.6f}",
                    flush=True,
                )
                print(
                    f"preservation penalty: {preservation_penalty:.6f}",
                    flush=True,
                )
                print(
                    f"preservation mode: {preservation_config.mode}",
                    flush=True,
                )
            if relation_preserve_this_update:
                print(f"Relation anchor layer: {layer}", flush=True)
                print(
                    f"relation_anchor_beta: {relation_anchor_config.beta}",
                    flush=True,
                )
                print(
                    f"relation_anchor_budget: {relation_artifact.requested_budget}",
                    flush=True,
                )
                print(
                    "effective selected anchor count: "
                    f"{relation_artifact.effective_budget}",
                    flush=True,
                )
                print(
                    "scanned candidate count: "
                    f"{relation_artifact.scanned_candidate_count}",
                    flush=True,
                )
                print(
                    f"matched relations: {relation_artifact.matched_relation_count}",
                    flush=True,
                )
                print(
                    "unmatched relations without usable candidates: "
                    f"{relation_artifact.unmatched_relation_count}",
                    flush=True,
                )
                print(
                    "relations unallocated due to budget: "
                    f"{relation_artifact.unallocated_relation_count}",
                    flush=True,
                )
                print(
                    "allocated anchors per relation min/mean/max: "
                    f"{relation_artifact.allocated_per_relation_min}/"
                    f"{relation_artifact.allocated_per_relation_mean:.3f}/"
                    f"{relation_artifact.allocated_per_relation_max}",
                    flush=True,
                )
                print(
                    "hardness score min/mean/max: "
                    f"{relation_artifact.hardness_min:.6f}/"
                    f"{relation_artifact.hardness_mean:.6f}/"
                    f"{relation_artifact.hardness_max:.6f}",
                    flush=True,
                )
                print(
                    f"neutral covariance norm: {neutral_cov_norm:.6f}",
                    flush=True,
                )
                print(
                    f"relation covariance norm: {relation_cov_norm:.6f}",
                    flush=True,
                )
                print(
                    "neutral preservation penalty: "
                    f"{neutral_preservation_penalty:.6f}",
                    flush=True,
                )
                print(
                    "relation preservation penalty: "
                    f"{relation_preservation_penalty:.6f}",
                    flush=True,
                )
                print(f"neutral_ratio: {neutral_ratio:.6f}", flush=True)
                print(f"relation_ratio: {relation_ratio:.6f}", flush=True)
                print(
                    f"final update norm: {torch.linalg.norm(upd_matrix).item():.6f}",
                    flush=True,
                )
            if calibration_diagnostics is not None:
                print(f"Calibration layer: {layer}", flush=True)
                print(
                    "response_calibration_alpha: "
                    f"{response_calibration_config.alpha}",
                    flush=True,
                )
                print(
                    "response_calibration_max_ratio: "
                    f"{response_calibration_config.max_ratio}",
                    flush=True,
                )
                print(
                    f"residual norm: {calibration_diagnostics.residual_norm:.6f}",
                    flush=True,
                )
                print(
                    "base response relative error: "
                    f"{calibration_diagnostics.base_relative_error:.6f}",
                    flush=True,
                )
                print(
                    "calibrated residual norm ratio: "
                    f"{calibration_diagnostics.calibrated_residual_ratio:.6f}",
                    flush=True,
                )
                print(
                    "final response relative error: "
                    f"{calibration_diagnostics.final_relative_error:.6f}",
                    flush=True,
                )
                print(
                    f"calibration clipped: {calibration_diagnostics.clipped}",
                    flush=True,
                )
                print(
                    "system factorization count: "
                    f"{calibration_diagnostics.factorization_count}",
                    flush=True,
                )
                print(
                    f"calibrated update norm: {torch.linalg.norm(upd_matrix).item():.6f}",
                    flush=True,
                )

            # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().to("cuda:0")
                deltas[weight_name] = upd_matrix

            # Clear GPU memory

            for x in [layer_ks, cur_zs, targets]:
                x.cpu()
                del x
            torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, _ in weights.items():
            nethook.get_parameter(model, k)[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    if return_preservation_updates:
        return deltas, pending_preservation_updates
    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )
def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = canonical_stats_model_name(model.config._name_or_path)
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats( # download
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            model_name=model_name,
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    try:
        return (
            torch.inverse(COV_CACHE[key].to("cuda:1")) if inv else COV_CACHE[key].to("cuda:1")
        )
    except:
        return (
            torch.inverse(COV_CACHE[key].to("cuda:0")) if inv else COV_CACHE[key].to("cuda:0")
        )
def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                ) # 用模型生成句子
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
