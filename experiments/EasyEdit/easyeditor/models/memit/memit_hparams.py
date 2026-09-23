from dataclasses import dataclass
from typing import List, Literal, Optional

from ...util.hparams import HyperParams
import yaml


@dataclass
class MEMITHyperParams(HyperParams):
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

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    alg_name: str
    device: int
    model_name: str
    stats_dir: str

    max_length: int = 40
    batch_size: int = 1
    model_parallel: bool = False
    bf16: bool = False

    # Paper-based original O-Edit soft loss at the final edited layer.
    # Disabled preserves the native/HN path; no O-Edit+ projection is applied.
    oedit_enabled: bool = False
    oedit_lambda_history: float = 50.0
    oedit_lambda_gradient: float = 50.0
    oedit_gradient_rank_per_edit: float = 1.0
    oedit_gradient_cache_path: Optional[str] = None
    oedit_loss_type: str = "mean_abs_cosine"
    oedit_log_path: Optional[str] = None

    # Residual Gain Regularization (RGR) modifies compute_z's latent objective;
    # MEMIT's closed-form outer solve remains unchanged unless an independent
    # opt-in outer-key ablation below is enabled.
    residual_gain_regularization: bool = False
    residual_gain_lambda: float = 0.1
    residual_gain_layers: Optional[List[int]] = None
    residual_gain_token_scope: str = "subject_last"
    residual_gain_subject_layers: Optional[List[int]] = None
    residual_gain_prompt_layers: Optional[List[int]] = None
    residual_gain_subject_lambda: Optional[float] = None
    residual_gain_prompt_lambda: Optional[float] = None
    residual_gain_margin: float = 0.0
    residual_gain_loss_type: str = "positive_l1"
    residual_gain_objective: str = "gain"
    residual_gain_alignment_weight: float = 1.0
    # Independent absolute auxiliary: phi_gamma(cos(H, F)), minimized at -1.
    # A zero lambda preserves the original RGR objective exactly.
    residual_gain_cosine_aux_lambda: float = 0.0
    residual_gain_cosine_aux_sharpness: float = 1.0
    residual_gain_efficacy_threshold: float = 0.0
    residual_gain_select_best: bool = False
    residual_gain_selection_threshold: float = 0.5
    residual_gain_early_stop_mode: str = "base"
    residual_gain_log_path: Optional[str] = None

    # Prefix-free stochastic outer-key ablation.  Exactly one canonical
    # prompt key is extracted per fact and replaced by k + epsilon; no
    # generated-prefix key is extracted or averaged.  The relative std gives
    # E||epsilon|| approximately equal to relative_std * ||k||.  A stable
    # per-layer/request seed makes the intervention independent of batching
    # and global RNG consumption.
    key_gaussian_noise_enabled: bool = False
    key_gaussian_noise_relative_std: float = 0.05
    key_gaussian_noise_seed: int = 42
    key_gaussian_noise_log_path: Optional[str] = None

    # Opt-in comparison methods.  Defaults preserve the original MEMIT path.
    sadr_regularization: bool = False
    sadr_lambda: float = 0.01
    sadr_attn_layers: Optional[List[int]] = None
    sadr_efficacy_threshold: float = 0.5
    sadr_log_path: Optional[str] = None

    encore_enabled: bool = False
    encore_mpes_top1_steps: int = 2
    encore_mpes_exclude_first_context: bool = True
    encore_norm_lambda: float = 20.0

    nse_enabled: bool = False
    nse_alpha: float = 2.5
    nse_upper_bound: int = 50
    nse_max_iterations: int = 3
    nse_neuron_threshold: float = 1.0
    nse_target_cache_dir: Optional[str] = None

    sphere_enabled: bool = False
    sphere_beta: float = 0.5
    sphere_alpha: float = 0.8
    official_baseline_log_path: Optional[str] = None

    # Norm Anchor Scaling (NAS).  NAS post-processes the optimized MLP value
    # before MEMIT's native closed-form outer solve; it is disabled by default.
    nas_enabled: bool = False
    nas_collect_stats: bool = False
    nas_anchor_path: Optional[str] = None
    nas_anchor_norm: Optional[float] = None
    nas_outlier_factor: float = 2.0
    nas_outlier_mode: str = "skip_delta"
    nas_log_path: Optional[str] = None

    # Endogenous-pivot MEMIT.  Native adjusted keys Q_l remain fixed while
    # value factors R_l are refined through the real transformer dynamics.
    # Disabled by default so existing MEMIT runs are bit-for-bit unaffected.
    endogenous_pivot_enabled: bool = False
    endogenous_pivot_mode: Literal["joint", "coordinate"] = "coordinate"
    # Optional layer ablation. None means every edit layer; e.g. [8] trains
    # only R_8 while still applying native MEMIT factors at the other layers.
    endogenous_pivot_trainable_layers: Optional[List[int]] = None
    # In coordinate mode this is the number of Adam steps per layer/visit. In
    # joint mode it is the total number of Adam steps; num_sweeps is ignored.
    endogenous_pivot_num_steps: int = 5
    endogenous_pivot_num_sweeps: int = 2
    endogenous_pivot_lr: float = 1e-2
    # Weight on covariance-normalized update energy. A positive value penalizes
    # (but does not rule out) last-layer concentration; inspect allocation logs.
    endogenous_pivot_l2_lambda: float = 1e-2
    endogenous_pivot_grad_clip_norm: float = 1.0
    endogenous_pivot_capture_trajectory: bool = False
    endogenous_pivot_log_path: Optional[str] = None
    endogenous_pivot_artifact_dir: Optional[str] = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'MEMIT') or print(f'MEMITHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
