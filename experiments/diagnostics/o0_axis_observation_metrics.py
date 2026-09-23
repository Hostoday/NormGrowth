"""CPU accounting for an opt-in O0-axis constrained selected target.

The optimizer shares one delta across all supplied contexts. Raw online
observations contain only the canonical request: projecting its realized O8
onto all optimizer axes does NOT reobserve every optimizer context.
"""
from __future__ import annotations
import numpy as np

PREFIX = "o0_axis_preservation_"
LAYER_PREFIX = "early_output_axes_"
STATE_ARRAYS = ("o0_context_vectors", "context_unit_axes", "reference", "reference_coefficients",
                "basis", "singular_values", "context_batch_indices", "lookup_indices", "resolved_lookup_indices")
TARGET_PHASES = ("canonical_inner_init", "returned_fp32_z", "virtual_bf16_injected", "actual_raw_O8")


def require(value, message):
    if not value:
        raise ValueError(message)


def state_arrays(state, width):
    """Copy a CPU NumPy state; tensor conversion is the observer's responsibility."""
    out = {PREFIX+k: np.asarray(state[k]).copy() for k in STATE_ARRAYS}
    axes = np.asarray(state["context_unit_axes"], dtype=np.float64)
    require(axes.ndim == 2 and axes.shape[1] == width and len(axes)>0, "O0 axes shape mismatch")
    count = len(axes)
    for key in ("o0_context_vectors", "reference"):
        require(np.asarray(state[key]).shape == (count, width), f"O0 {key} shape mismatch")
    require(np.asarray(state["reference_coefficients"]).shape == (count,), "O0 reference coefficient shape mismatch")
    for key in ("context_batch_indices", "lookup_indices", "resolved_lookup_indices"):
        a=np.asarray(state[key]);require(a.shape == (count,) and np.issubdtype(a.dtype,np.integer), f"O0 {key} shape/type mismatch")
    for key in STATE_ARRAYS:
        require(np.isfinite(out[PREFIX+key]).all(), f"O0 {key} nonfinite")
    require(np.allclose(np.linalg.norm(axes,axis=-1),1,atol=2e-6,rtol=0), "O0 axes must be unit vectors")
    vectors = np.asarray(state["o0_context_vectors"],dtype=np.float64)
    norms=np.linalg.norm(vectors,axis=-1)
    require(np.all(norms>0) and np.allclose(axes,vectors/norms[:,None],atol=2e-6,rtol=2e-6), "O0 axis/raw-vector mismatch")
    reference=np.asarray(state["reference"],dtype=np.float64)
    require(np.allclose(np.sum(reference*axes,axis=-1),state["reference_coefficients"],atol=2e-5,rtol=2e-5), "O0 reference coefficient mismatch")
    canonical=np.flatnonzero(np.asarray(state["context_batch_indices"])==0)
    require(len(canonical)==1,"O0 state requires exactly one canonical context batch row0")
    axis_layers=list(state.get("axis_layers",[0]))
    require(axis_layers in ([0],[0,1,2,3]),"unsupported preserved output-axis layers")
    rank=int(state["rank"])
    require(0<rank<=min(count*len(axis_layers),width) and np.asarray(state["basis"]).shape==(width,rank), "O0 basis/rank mismatch")
    out[PREFIX+"rank"]=np.asarray(rank,dtype=np.int64)
    out[PREFIX+"canonical_context_index"]=np.asarray(canonical[0],dtype=np.int64)
    if len(axis_layers)>1:
        layer_axes=np.asarray(state["context_layer_unit_axes"],dtype=np.float64)
        layer_vectors=np.asarray(state["context_layer_output_vectors"],dtype=np.float64)
        layer_coeff=np.asarray(state["context_layer_reference_coefficients"],dtype=np.float64)
        require(layer_axes.shape==layer_vectors.shape==(count,len(axis_layers),width),"layer axes/vector shape mismatch")
        require(layer_coeff.shape==(count,len(axis_layers)),"layer reference coefficient shape mismatch")
        require(np.isfinite(layer_vectors).all() and np.isfinite(layer_axes).all() and np.isfinite(layer_coeff).all(),"nonfinite early output axes")
        layer_norms=np.linalg.norm(layer_vectors,axis=-1)
        require(np.all(layer_norms>0) and np.allclose(layer_axes,layer_vectors/layer_norms[...,None],atol=2e-6,rtol=2e-6),"layer axis/raw-vector mismatch")
        require(np.array_equal(layer_axes[:,0],axes) and np.array_equal(layer_vectors[:,0],vectors),"legacy O0 slice differs from layer0")
        require(np.allclose(np.einsum('cd,ckd->ck',reference,layer_axes),layer_coeff,atol=2e-5,rtol=2e-5),"layer reference coefficient mismatch")
        out[LAYER_PREFIX+"axis_layers"]=np.asarray(axis_layers,dtype=np.int64)
        for key in ("context_layer_unit_axes","context_layer_output_vectors","context_layer_reference_coefficients"):
            out[LAYER_PREFIX+key]=np.asarray(state[key]).copy()
    return out


def observation_metrics(arrays):
    """Recompute all numerical observations from stored vectors in float64."""
    axes=np.asarray(arrays[PREFIX+"context_unit_axes"],dtype=np.float64)
    c=int(arrays[PREFIX+"canonical_context_index"])
    init=np.asarray(arrays["target_init"],dtype=np.float64)
    z=np.asarray(arrays["optimized_z"],dtype=np.float64)
    targets=np.stack([init,z,arrays["virtual_injected_subject_output"],arrays["actual_subject_output"]]).astype(np.float64)
    reference=np.asarray(arrays[PREFIX+"reference"],dtype=np.float64)
    require(np.array_equal(init,reference[c]),"canonical inner init differs from O0 preserver reference")
    require(int(arrays[PREFIX+"resolved_lookup_indices"][c])==int(arrays["token_positions"][0]),"canonical O0 lookup differs from raw subject token")
    require(np.array_equal(arrays["virtual_o"][:,:4],arrays["actual_o"][:,:4]),"O0-to-O3 virtual/actual invariant failed")
    coeff=targets@axes.T
    result={PREFIX+"target_phase_order":np.asarray(TARGET_PHASES),
            PREFIX+"target_vectors_axis_coefficients":coeff,
            PREFIX+"target_vectors_axis_drifts_from_canonical_init":coeff-coeff[0],
            PREFIX+"optimizer_delta_axis_dot":np.asarray(arrays["optimizer_delta"],dtype=np.float64)@axes.T,
            PREFIX+"returned_delta_axis_dot":(z-init)@axes.T,
            PREFIX+"injection_rounding_axis_dot":(targets[2]-z)@axes.T,
            PREFIX+"actual_minus_injected_axis_dot":(targets[3]-targets[2])@axes.T,
            PREFIX+"raw_native_minus_inner_init_axis_dot":(np.asarray(arrays["virtual_native_subject_output"],dtype=np.float64)-init)@axes.T}
    raw_o0=np.asarray(arrays["virtual_o"][:,0],dtype=np.float64)
    raw_norm=np.linalg.norm(raw_o0,axis=-1)
    require(np.all(raw_norm>0),"zero raw O0 norm")
    raw_axes=raw_o0/raw_norm[:,None]
    result[PREFIX+"raw_query_unit_axes"]=raw_axes
    result[PREFIX+"raw_vs_optimizer_canonical_axis_cosine"]=np.asarray(raw_axes[0]@axes[c])
    result[PREFIX+"raw_native_O8_coefficients"]=np.sum(np.asarray(arrays["virtual_native_target_layer_outputs"],dtype=np.float64)*raw_axes,axis=-1)
    for phase in ("virtual","actual"):
        for node in ("o","a","m"):
            result[PREFIX+f"raw_{phase}_{node}_coefficients"]=np.einsum("pld,pd->pl",np.asarray(arrays[f"{phase}_{node}"],dtype=np.float64),raw_axes)
    for key,value in result.items():
        if key!=PREFIX+"target_phase_order":require(np.isfinite(value).all(),f"nonfinite O0 observation: {key}")
    scale=max(1.,float(np.linalg.norm(z)),float(np.linalg.norm(init)))
    returned_error=float(np.max(np.abs(result[PREFIX+"returned_delta_axis_dot"])))
    metadata=dict(schema_version=1,enabled=True,reference_semantics="current_pre_edit",
                  axis_scope="all supplied injected optimizer contexts; canonical raw observer is separate",
                  canonical_context_index=c,n_contexts=len(axes),rank=int(arrays[PREFIX+"rank"]),
                  measurement_precision="float64 accounting from exact stored FP32 expanded BF16/FP32 vectors",
                  target_phase_order=list(TARGET_PHASES),
                  returned_delta_axis_dot_abs_max=returned_error,
                  returned_delta_axis_dot_relative_to_target_norm=returned_error/scale,
                  optimizer_constraint_relative_tolerance=5e-5,
                  optimizer_constraint_within_numerical_tolerance=returned_error/scale<=5e-5,
                  actual_realization_is_not_hard_constrained=True,
                  all_axis_realization_semantics="One canonical raw-request O8 vector projected onto every optimizer axis, not actual forwards of all optimizer contexts",
                  raw_trajectory_semantics="Each query's own current pre-edit canonical O0 unit axis; subject and prompt axes differ",
                  raw_virtual_m_semantics="Native branch M, excluding external O8 target injection")
    if LAYER_PREFIX+"axis_layers" in arrays:
        expanded,expanded_metadata=layer_observation_metrics(arrays,targets,init,z,scale,c)
        result.update(expanded)
        metadata["early_output_axes"]=expanded_metadata
    return result,metadata


def layer_observation_metrics(arrays,targets,init,z,scale,canonical):
    """All C×K constraints; one canonical realized vector, K raw query axes."""
    layers=np.asarray(arrays[LAYER_PREFIX+'axis_layers'])
    require(np.array_equal(layers,[0,1,2,3]),"expanded observer requires ordered O0..O3 axes")
    axes=np.asarray(arrays[LAYER_PREFIX+'context_layer_unit_axes'],dtype=np.float64)
    require(axes.shape==(len(arrays[PREFIX+'context_unit_axes']),4,len(init)),"expanded axis layout differs")
    coeff=np.einsum('fd,ckd->fck',targets,axes)
    delta_dots=np.einsum('d,ckd->ck',z-init,axes)
    raw_vectors=np.asarray(arrays['virtual_o'][:,layers],dtype=np.float64)
    norms=np.linalg.norm(raw_vectors,axis=-1)
    require(np.all(norms>0),"zero raw early O norm")
    raw_axes=raw_vectors/norms[...,None]
    out={LAYER_PREFIX+'target_vectors_axis_coefficients':coeff,
         LAYER_PREFIX+'target_vectors_axis_drifts_from_canonical_init':coeff-coeff[0],
         LAYER_PREFIX+'returned_delta_axis_dot':delta_dots,
         LAYER_PREFIX+'optimizer_delta_axis_dot':np.einsum('d,ckd->ck',np.asarray(arrays['optimizer_delta'],dtype=np.float64),axes),
         LAYER_PREFIX+'injection_rounding_axis_dot':np.einsum('d,ckd->ck',targets[2]-z,axes),
         LAYER_PREFIX+'actual_minus_injected_axis_dot':np.einsum('d,ckd->ck',targets[3]-targets[2],axes),
         LAYER_PREFIX+'raw_native_minus_inner_init_axis_dot':np.einsum('d,ckd->ck',np.asarray(arrays['virtual_native_subject_output'],dtype=np.float64)-init,axes),
         LAYER_PREFIX+'raw_query_unit_axes':raw_axes,
         LAYER_PREFIX+'raw_vs_optimizer_canonical_axis_cosine':np.einsum('kd,kd->k',raw_axes[0],axes[canonical]),
         LAYER_PREFIX+'raw_native_O8_coefficients':np.einsum('pd,pkd->pk',np.asarray(arrays['virtual_native_target_layer_outputs'],dtype=np.float64),raw_axes)}
    for phase in ('virtual','actual'):
        for node in ('o','a','m'):
            out[LAYER_PREFIX+f'raw_{phase}_{node}_coefficients']=np.einsum('pld,pkd->pkl',np.asarray(arrays[f'{phase}_{node}'],dtype=np.float64),raw_axes)
    require(all(np.isfinite(v).all() for v in out.values()),"nonfinite expanded early-axis observations")
    error=float(np.max(np.abs(delta_dots)))
    meta=dict(axis_layers=layers.tolist(),n_contexts=len(axes),n_constraints=int(np.prod(axes.shape[:2])),
              target_coefficient_layout='target_phase, optimizer_context, axis_layer',raw_coefficient_layout='query_position, axis_layer, destination_layer',
              returned_delta_axis_dot_abs_max=error,returned_delta_axis_dot_relative_to_target_norm=error/scale,
              optimizer_constraint_relative_tolerance=5e-5,optimizer_constraint_within_numerical_tolerance=error/scale<=5e-5,
              actual_semantics='One canonical raw O8 projected onto all optimizer context/layer axes; not actual forwards of all contexts',
              native_M_excludes_virtual_O8_injection=True)
    return out,meta
