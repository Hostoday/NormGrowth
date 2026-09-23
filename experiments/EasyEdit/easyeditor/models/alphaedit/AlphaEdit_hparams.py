from dataclasses import dataclass
from typing import List, Literal, Optional

from ...util.hparams import HyperParams
import yaml


@dataclass
class AlphaEditHyperParams(HyperParams):
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
    nullspace_threshold: float
    L2: float
    alg_name: str
    device: int
    model_name: str
    stats_dir: str
    P_loc: str

    max_length: int = 40
    batch_size: int = 1
    model_parallel: bool = False
    bf16: bool = False
    target_alpha: float = 0.8
    pl_skip_above_target: bool = False
    skip_prob: float = 1.0
    rewrite_kl_lambda: float = 1.2
    use_kl : bool = True

    # Paper-based original O-Edit soft loss at the final edited layer. Disabled
    # preserves the existing HN/native path; O-Edit+ projection is not included.
    oedit_enabled: bool = False
    oedit_lambda_history: float = 50.0
    oedit_lambda_gradient: float = 50.0
    oedit_gradient_rank_per_edit: float = 1.0
    oedit_gradient_cache_path: Optional[str] = None
    oedit_loss_type: str = "mean_abs_cosine"
    oedit_log_path: Optional[str] = None

    # Residual Gain Regularization (RGR).  The regularizer acts only while
    # optimizing v*; AlphaEdit's null-space projected outer solve is unchanged.
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
    # Opt-in HN recovery experiments.  ``target_ratio=1`` retains the exact
    # historical rho-minus-one arithmetic; values above one relax the
    # one-sided ceiling to current_rho <= R * reference_rho.
    residual_gain_target_ratio: float = 1.0
    # Independent absolute auxiliary: phi_gamma(cos(H, F)), minimized at -1.
    # A zero lambda preserves the original RGR objective exactly.
    residual_gain_cosine_aux_lambda: float = 0.0
    residual_gain_cosine_aux_sharpness: float = 1.0
    residual_gain_efficacy_threshold: float = 0.0
    residual_gain_select_best: bool = False
    residual_gain_selection_threshold: float = 0.5
    residual_gain_early_stop_mode: str = "base"
    residual_gain_log_path: Optional[str] = None

    # HN+attn: preserve normalized L8 subject-output coordinates in the
    # current pre-edit A0-A4 span across the six canonical rewrite contexts.
    early_attention_preservation_enabled: bool = False
    early_attention_preservation_lambda: float = 1.0
    early_attention_preservation_layers: Optional[List[int]] = None
    early_attention_preservation_reference: str = "current_pre_edit"
    early_attention_preservation_log_path: Optional[str] = None

    # Hard early-O-axis preservation during z optimization.  The same effective
    # delta is orthogonal to the span of the frozen selected O unit directions at all
    # injected context/lookup positions, preserving each pre-edit O8 signed
    # coefficient.  The native outer AlphaEdit solve remains unchanged.
    o0_axis_preservation_enabled: bool = False
    # None preserves the original single-O0 experiment; [0,1,2,3] protects
    # every raw block-output direction at those layers and all injected rows.
    o0_axis_preservation_layers: Optional[List[int]] = None
    o0_axis_preservation_reference: str = "current_pre_edit"
    o0_axis_preservation_scope: str = "all_injected_contexts"
    o0_axis_preservation_svd_rtol: float = 1.0e-6
    o0_axis_preservation_log_path: Optional[str] = None

    # Observer only. A runner supplies the callback; losses and outer solves
    # are unchanged. Only the finally selected/returned latent is published.
    analysis_capture_virtual_actual: bool = False

    # Add a frozen, target-blind optimization paraphrase family to compute_z.
    # The official ZsRE ``rephrase_prompt`` is evaluation-only and is rejected
    # if it is reused as ``optimization_paraphrase``.
    hn_recovery_paraphrase_aware: bool = False
    hn_recovery_edit_family_weight: float = 1.0
    hn_recovery_para_family_weight: float = 0.5

    # Per-request difficulty adaptation.  Each ceiling is solved from a fresh
    # delta=0/Adam state; the first candidate whose canonical unprefixed target
    # geomean probability reaches the threshold is committed.
    hn_recovery_adaptive_rho: bool = False
    hn_recovery_adaptive_threshold: float = 0.5
    hn_recovery_adaptive_ladder: Optional[List[float]] = None

    # Selected per-edit scalar record.  Attempts are nested in the selected
    # row so retries never overwrite each other.
    hn_recovery_scalar_log_path: Optional[str] = None

    # Opt-in causal ablation for separating target formation from preservation.
    # ``legacy`` is byte-for-byte the historical control path. ``fixed_stop``
    # stops once every supervised rewrite token clears the requested margin;
    # ``delayed_rgr`` then runs a fixed number of ordinary NLL+RGR updates;
    # ``two_stage`` replaces NLL by a margin-floor hinge during refinement.
    inner_margin_schedule: str = "legacy"
    inner_target_margin_threshold: float = 0.0
    inner_margin_refinement_steps: int = 0
    inner_margin_hinge_weight: float = 1.0
    inner_margin_schedule_log_path: Optional[str] = None

    # Exact same-target multi-key compression over the raw context keys that
    # AlphaEdit already extracts (currently one original plus five generated
    # prefixes).  Disabled by default so existing YAMLs and runs retain the
    # native mean-key solve and cache semantics.
    context_multikey_enabled: bool = False
    context_multikey_log_path: Optional[str] = None
    # The current controlled experiment must use the complete existing layout:
    # one identity/original template and all five generated-prefix templates.
    # The analyzers set this to [1, 5] so a missing/reduced context fails closed.
    context_multikey_expected_group_sizes: Optional[List[int]] = None

    # Prefix-free stochastic key ablation for the outer solve.  Exactly one
    # canonical prompt key is extracted per fact and replaced by k + epsilon;
    # no context/prefix key is extracted or averaged.  ``relative_std`` is
    # dimensionless: E||epsilon|| is approximately relative_std * ||k||.
    # The seed is deterministically split by layer and request identity so the
    # committed cache uses the same noise direction as the corresponding
    # solve.  This mode is mutually exclusive with context multi-key.
    key_gaussian_noise_enabled: bool = False
    key_gaussian_noise_relative_std: float = 0.05
    key_gaussian_noise_seed: int = 42
    key_gaussian_noise_log_path: Optional[str] = None

    # Strict joint local-tangent allocation for the outer write.  This is an
    # independent axis from context multi-key retrieval.  It captures the
    # clean block output at every edited layer, decomposes the initial z*
    # error once under J_l ~= I, and never gives tangent-null slack to the
    # terminal layer.  The first causal experiment is deliberately bs=1.
    tangent_layer_allocation_enabled: bool = False
    tangent_layer_allocation_rcond: Optional[float] = None
    tangent_layer_allocation_max_relative_slack: Optional[float] = None
    tangent_layer_allocation_log_path: Optional[str] = None

    # Opt-in comparison methods.  Defaults preserve the original AlphaEdit path.
    sadr_regularization: bool = False
    sadr_lambda: float = 0.01
    sadr_attn_layers: Optional[List[int]] = None
    sadr_efficacy_threshold: float = 0.5
    sadr_log_path: Optional[str] = None

    encore_enabled: bool = False
    encore_mpes_top1_steps: int = 2
    encore_mpes_exclude_first_context: bool = True
    encore_norm_lambda: float = 0.0

    nse_enabled: bool = False
    nse_alpha: float = 2.5
    nse_upper_bound: int = 50
    nse_max_iterations: int = 3
    nse_neuron_threshold: float = 1.0
    nse_target_cache_dir: Optional[str] = None

    sphere_enabled: bool = False
    sphere_beta: float = 0.5
    sphere_alpha: float = 0.5
    official_baseline_log_path: Optional[str] = None

    # Norm Anchor Scaling (NAS).  The anchored target is passed to AlphaEdit's
    # native null-space projected outer solve.
    nas_enabled: bool = False
    nas_collect_stats: bool = False
    nas_anchor_path: Optional[str] = None
    nas_anchor_norm: Optional[float] = None
    nas_outlier_factor: float = 2.0
    nas_outlier_mode: str = "skip_delta"
    nas_log_path: Optional[str] = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'AlphaEdit') or print(f'AlphaEditHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
