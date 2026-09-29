"""State and numerics for fixed-beta PRCMET preservation and calibration.

The historical covariance convention is an EMA of the unnormalised per-batch
Gram matrix ``K @ K.T``. Updates are staged so the covariance advances only
after the corresponding model weight writes succeed. Single-shot anchor
covariances are implemented separately in :mod:`prcmet.anchors`.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch


StateKey = Tuple[str, str]
COVARIANCE_CONVENTION = "raw_kkT_ema"
STATE_VERSION = 2


@dataclass(frozen=True)
class PreservationConfig:
    enabled: bool
    beta: float
    rho: float
    mode: str = "history"

    @classmethod
    def from_hparams(cls, hparams: Any) -> "PreservationConfig":
        config = cls(
            enabled=getattr(hparams, "preservation_constraint", False),
            beta=float(getattr(hparams, "beta_preserve", 0.1)),
            rho=float(getattr(hparams, "preserve_rho", 0.95)),
            mode=getattr(hparams, "preservation_mode", "history"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("preservation_constraint must be a boolean.")
        if not math.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta_preserve must be finite and non-negative.")
        if not math.isfinite(self.rho) or not 0.0 <= self.rho <= 1.0:
            raise ValueError("preserve_rho must be finite and between 0 and 1.")
        if self.mode not in {"history", "anchors"}:
            raise ValueError("preservation_mode must be 'history' or 'anchors'.")


@dataclass(frozen=True)
class ResponseCalibrationConfig:
    """Configuration for bounded first-order response calibration."""

    enabled: bool
    alpha: float
    max_ratio: float
    solve_chunk_size: int = 256

    @classmethod
    def from_hparams(cls, hparams: Any) -> "ResponseCalibrationConfig":
        config = cls(
            enabled=getattr(hparams, "response_calibration", False),
            alpha=float(getattr(hparams, "response_calibration_alpha", 0.5)),
            max_ratio=float(
                getattr(hparams, "response_calibration_max_ratio", 1.25)
            ),
            solve_chunk_size=getattr(
                hparams, "response_calibration_solve_chunk_size", 256
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("response_calibration must be a boolean.")
        if not math.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                "response_calibration_alpha must be finite and between 0 and 1."
            )
        if not math.isfinite(self.max_ratio) or self.max_ratio < 1.0:
            raise ValueError(
                "response_calibration_max_ratio must be finite and at least 1."
            )
        if (
            isinstance(self.solve_chunk_size, bool)
            or not isinstance(self.solve_chunk_size, int)
            or self.solve_chunk_size <= 0
        ):
            raise ValueError(
                "response_calibration_solve_chunk_size must be a positive integer."
            )


@dataclass(frozen=True)
class ResponseCalibrationDiagnostics:
    residual_norm: float
    base_relative_error: float
    calibrated_residual_ratio: float
    final_relative_error: float
    clipped: bool
    factorization_count: int


@dataclass(frozen=True)
class FactorizedRightSystem:
    lu: torch.Tensor
    pivots: torch.Tensor


def factorize_right_system(system_matrix: torch.Tensor) -> FactorizedRightSystem:
    """Factor ``A.T`` once for repeated right-side solves against ``A^-1``."""

    if system_matrix.ndim != 2 or system_matrix.shape[0] != system_matrix.shape[1]:
        raise ValueError("system_matrix must be square.")
    if not system_matrix.dtype.is_floating_point:
        raise ValueError("system_matrix must use a floating-point dtype.")
    lu, pivots = torch.linalg.lu_factor(system_matrix.T)
    return FactorizedRightSystem(lu=lu, pivots=pivots)


@dataclass(frozen=True)
class PendingPreservationUpdate:
    key: StateKey
    key_cov_cpu: torch.Tensor
    rho: float


def right_solve_update(
    residual: torch.Tensor,
    keys: torch.Tensor,
    system_matrix: torch.Tensor,
) -> torch.Tensor:
    """Return ``residual @ keys.T @ inv(system_matrix)`` via one solve."""

    numerator = residual @ keys.T
    return torch.linalg.solve(system_matrix.T, numerator.T).T


def bounded_response_calibrated_update(
    residual: torch.Tensor,
    keys: torch.Tensor,
    factorization: FactorizedRightSystem,
    config: ResponseCalibrationConfig,
) -> Tuple[torch.Tensor, ResponseCalibrationDiagnostics]:
    """Compute one joint calibrated update with one factorization and chunked solves."""

    config.validate()
    if residual.ndim != 2 or keys.ndim != 2:
        raise ValueError("residual and keys must both be matrices.")
    if residual.shape[1] != keys.shape[1]:
        raise ValueError("residual and keys must contain the same number of columns.")
    if factorization.lu.shape != (keys.shape[0], keys.shape[0]):
        raise ValueError("factorization size must match the keys' first dimension.")
    if not (
        residual.device == keys.device == factorization.lu.device
        and residual.dtype == keys.dtype == factorization.lu.dtype
    ):
        raise ValueError("residual, keys, and factorization must share dtype and device.")
    if not residual.dtype.is_floating_point:
        raise ValueError("response calibration requires floating-point tensors.")

    chunk_size = config.solve_chunk_size
    num_columns = keys.shape[1]

    # Every key block reuses the one supplied factorization; no full
    # Q = K.T A^-1 and no N x N response matrix are materialized.
    lu, pivots = factorization.lu, factorization.pivots
    factorization_count = 1

    base_update = torch.zeros(
        (residual.shape[0], keys.shape[0]),
        dtype=residual.dtype,
        device=residual.device,
    )
    for start in range(0, num_columns, chunk_size):
        end = min(start + chunk_size, num_columns)
        key_chunk = keys[:, start:end]
        residual_chunk = residual[:, start:end]
        right_factor_chunk = torch.linalg.lu_solve(
            lu, pivots, key_chunk
        ).T
        base_update.add_(residual_chunk @ right_factor_chunk)

    residual_norm = torch.linalg.vector_norm(residual)
    raw_norm_squared = torch.zeros((), dtype=residual.dtype, device=residual.device)
    base_error_squared = torch.zeros_like(raw_norm_squared)
    raw_update = torch.zeros_like(base_update)
    for start in range(0, num_columns, chunk_size):
        end = min(start + chunk_size, num_columns)
        key_chunk = keys[:, start:end]
        residual_chunk = residual[:, start:end]
        right_factor_chunk = torch.linalg.lu_solve(
            lu, pivots, key_chunk
        ).T
        base_error_chunk = residual_chunk - base_update @ key_chunk
        raw_residual_chunk = residual_chunk + config.alpha * base_error_chunk
        raw_update.add_(raw_residual_chunk @ right_factor_chunk)
        raw_norm_squared.add_(torch.sum(raw_residual_chunk.square()))
        base_error_squared.add_(torch.sum(base_error_chunk.square()))
    del base_update

    raw_norm = torch.sqrt(raw_norm_squared)
    epsilon = torch.finfo(residual.dtype).eps
    denominator = torch.clamp(residual_norm, min=epsilon)
    max_norm = config.max_ratio * residual_norm
    clipped = bool((raw_norm > max_norm).item())
    if clipped:
        calibration_scale = max_norm / torch.clamp(raw_norm, min=epsilon)
    else:
        calibration_scale = torch.ones(
            (), dtype=residual.dtype, device=residual.device
        )

    raw_update.mul_(calibration_scale)
    update = raw_update

    final_error_squared = torch.zeros_like(raw_norm_squared)
    for start in range(0, num_columns, chunk_size):
        end = min(start + chunk_size, num_columns)
        final_error_chunk = residual[:, start:end] - update @ keys[:, start:end]
        final_error_squared.add_(torch.sum(final_error_chunk.square()))
    if not torch.isfinite(update).all().item():
        raise FloatingPointError("Bounded response calibration produced a non-finite update.")

    diagnostics = ResponseCalibrationDiagnostics(
        residual_norm=residual_norm.item(),
        base_relative_error=(torch.sqrt(base_error_squared) / denominator).item(),
        calibrated_residual_ratio=(
            calibration_scale * raw_norm / denominator
        ).item(),
        final_relative_error=(torch.sqrt(final_error_squared) / denominator).item(),
        clipped=clipped,
        factorization_count=factorization_count,
    )
    return update, diagnostics


class PreservationState:
    """Run-local, layer-isolated covariance history with staged commits."""

    def __init__(self) -> None:
        self.covariances: Dict[StateKey, torch.Tensor] = {}
        self.batch_index = 0

    def reset(self) -> None:
        self.covariances.clear()
        self.batch_index = 0

    def old_covariance_on(
        self,
        key: StateKey,
        reference: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        covariance = self.covariances.get(key)
        if covariance is None:
            return None
        if covariance.shape != reference.shape:
            raise ValueError(
                f"Incompatible preservation covariance shape for {key}: checkpoint/cache "
                f"has {tuple(covariance.shape)}, current mom2 has {tuple(reference.shape)}."
            )
        if covariance.dtype != reference.dtype:
            raise ValueError(
                f"Incompatible preservation covariance dtype for {key}: checkpoint/cache "
                f"has {covariance.dtype}, current mom2 has {reference.dtype}."
            )
        return covariance.to(device=reference.device, dtype=reference.dtype)

    def stage(
        self,
        key: StateKey,
        current_gram: torch.Tensor,
        mom2_covariance: torch.Tensor,
        config: PreservationConfig,
    ) -> Tuple[torch.Tensor, PendingPreservationUpdate]:
        """Read the old covariance and stage the current Gram for later commit."""

        config.validate()
        old_covariance = self.old_covariance_on(key, mom2_covariance)
        covariance_for_solve = (
            torch.zeros_like(mom2_covariance)
            if old_covariance is None
            else old_covariance
        )
        pending = PendingPreservationUpdate(
            key=key,
            key_cov_cpu=current_gram.detach().to(
                device="cpu", dtype=mom2_covariance.dtype
            ),
            rho=config.rho,
        )
        return covariance_for_solve, pending

    def commit(self, updates: Iterable[PendingPreservationUpdate]) -> None:
        updates = list(updates)
        seen = set()
        for update in updates:
            if update.key in seen:
                raise ValueError(
                    f"A preservation transaction contains duplicate layer {update.key}."
                )
            seen.add(update.key)
            covariance = self.covariances.get(update.key)
            if covariance is not None and (
                covariance.shape != update.key_cov_cpu.shape
                or covariance.dtype != update.key_cov_cpu.dtype
            ):
                raise ValueError(
                    f"Pending covariance update is incompatible with state for {update.key}."
                )
            if not 0.0 <= update.rho <= 1.0:
                raise ValueError(f"Invalid covariance EMA coefficient for {update.key}.")

        for update in updates:
            covariance = self.covariances.get(update.key)
            if covariance is None:
                covariance = torch.zeros_like(update.key_cov_cpu, device="cpu")
            covariance.mul_(update.rho).add_(
                update.key_cov_cpu, alpha=1.0 - update.rho
            )
            self.covariances[update.key] = covariance
        if updates:
            self.batch_index += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "covariance_convention": COVARIANCE_CONVENTION,
            "batch_index": self.batch_index,
            "covariances": {
                key: value.detach().to("cpu").clone()
                for key, value in self.covariances.items()
            },
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        expected_model_name: Optional[str] = None,
    ) -> None:
        if state.get("version") != STATE_VERSION:
            raise ValueError(
                f"Unsupported fixed preservation state version: {state.get('version')!r}."
            )
        if state.get("covariance_convention") != COVARIANCE_CONVENTION:
            raise ValueError(
                "Incompatible preservation covariance convention: "
                f"{state.get('covariance_convention')!r}; expected "
                f"{COVARIANCE_CONVENTION!r}."
            )

        covariances: Dict[StateKey, torch.Tensor] = {}
        for raw_key, covariance in state.get("covariances", {}).items():
            if not isinstance(raw_key, (tuple, list)):
                raise ValueError(f"Invalid preservation state key: {raw_key!r}.")
            key = tuple(raw_key)
            if len(key) != 2 or not all(isinstance(part, str) for part in key):
                raise ValueError(f"Invalid preservation state key: {raw_key!r}.")
            if expected_model_name is not None and key[0] != expected_model_name:
                raise ValueError(
                    f"Checkpoint model {key[0]!r} does not match expected model "
                    f"{expected_model_name!r}."
                )
            if not isinstance(covariance, torch.Tensor):
                raise ValueError(f"Covariance for {key} is not a tensor.")
            if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                raise ValueError(f"Covariance for {key} is not square.")
            covariances[key] = covariance.detach().to("cpu").clone()

        batch_index = int(state.get("batch_index", 0))
        if batch_index < 0:
            raise ValueError("Preservation batch_index must be non-negative.")
        self.reset()
        self.covariances.update(covariances)
        self.batch_index = batch_index
