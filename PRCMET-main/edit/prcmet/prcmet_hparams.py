from dataclasses import dataclass
from typing import List, Optional
from typing_extensions import Literal

from util.hparams import HyperParams


@dataclass
class PRCMETHyperParams(HyperParams):
    # Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float
    nll_loss_factor: float

    # Module templates
    rewrite_module_tmp: str
    rewrite_module_tmps: List[str]
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    # Lightweight soft preservation constraint. Defaults preserve original PRCMET.
    preservation_constraint: bool = False
    preservation_mode: Literal["history", "anchors"] = "history"
    beta_preserve: float = 0.1
    preserve_rho: float = 0.95

    # Request-conditioned pre-edit anchors for single-shot preservation.
    anchor_templates: Optional[List[str]] = None
    anchor_max_count: int = 10000
    anchor_extraction_chunk_size: int = 128
    anchor_covariance_dtype: str = "float32"

    # Fixed-beta hard anchors selected from held-out facts with the same relation.
    relation_anchor_preservation: bool = False
    relation_anchor_beta: float = 0.15
    relation_anchor_budget: int = 1024

    # Optional bounded compensation for update attenuation on the current keys.
    response_calibration: bool = False
    response_calibration_alpha: float = 0.5
    response_calibration_max_ratio: float = 1.25
    response_calibration_solve_chunk_size: int = 256
