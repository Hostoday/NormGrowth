# Measurement input schema

The analysis commands read measurements from a flat directory supplied with `--data-dir`. No measurement tables or published reference results are bundled. Collect the required measurements from your model runs and convert them to the schemas below. The experimental runners write stage-specific artifacts; the repository does not provide a single exporter that converts every runner's output into all of these tables.

The commands implement the paper's experimental design: two models, two datasets, two editors, five methods, one canonical edit order, and 1,000 edits per configuration. Several checks require this complete design. Adapting the analysis to a different design also requires changing those checks and the corresponding analysis selections.

## Shared fields and units

| Fields | Meaning and accepted values |
|---|---|
| `model` | `Llama` or `GPT-2 XL` |
| `dataset` | `zsRE` or `CounterFact` |
| `editor` | `MEMIT` or `AlphaEdit` |
| `method` | `Native`, `NAS`, `ENCORE`, `SPHERE`, or `SADR` |
| `order_id` | `canonical` |
| `edit_index` | Integer identifying the edit within a configuration; consistent across RQ1 tables |
| `case_id` | Request identifier, read as a string for paired analyses; preserve identifiers and case order |
| `edit_count` | Cumulative number of edits |
| `trajectory_id` | Unique string for one model/dataset/editor/method/order combination |
| `row_id` | Unique checkpoint identifier shared by checkpoint and endpoint tables |

Below, **configuration fields** means `model,dataset,editor,method,order_id`; **state fields** adds `edit_count`. Ratios, cosines, and case accuracies are dimensionless. `_percent` scores use a 0–100 scale; `_pp` fields and the three `delta_*` fields in `intervention_doses.csv` are percentage-point differences. Preserve unrounded values. Boolean fields use `True`/`False` or `1`/`0`; blank numeric values represent undefined measurements, where explicitly allowed below.

## Required statistical inputs

`analysis.reproduce` requires the following seven CSV files in the input directory.

| File | Row identity | Required measurement columns |
|---|---|---|
| `rq1_per_edit.csv` | Configuration fields + `edit_index` | `returned_target_ratio`, `actual_post_pre_ratio` |
| `rq1_vectors_per_edit.csv` | Configuration fields + `edit_index,case_id` | Norm, vector, and availability fields listed below |
| `rq1_reference_sensitivity_per_edit.csv` | Same edit identities as the vector table | Paired-reference fields listed below |
| `checkpoints.csv` | State fields | `trajectory_id`, `locality_percent`, and both contexts' geometry summaries listed below; add `row_id` for plotting |
| `intervention_doses.csv` | State fields + `dose_fraction` | `delta_LOC`, `delta_EFF_TF`, `delta_GEN_TF`, `LOC_up_no_edit_loss`, `LOC_up_both_losses_at_most_1pp` |
| `same_norm_conditions.csv` | State fields + `dose_fraction,operation,endpoint` | `scope`, `n_cases`, `delta_pp` |
| `rq3_selected_case_scores.csv` | State fields + `dose_fraction,family,condition,case_id` | `eligible_at_both_doses`, `endpoint_value`, `n_target_tokens`, `semantic_accuracy`, `terminator_accuracy` |

### RQ1: per-edit norm and vector measurements

Each RQ1 table contains one row per edit for every configuration. For a single edit, let `z` be the optimized target, `h_init` the initialization used by the target optimizer, and `h_pre`/`h_post` the realized representations before/after the weight update. These per-edit references differ from the fixed unedited Base reference used in RQ2.

The vector table needs the following fields in addition to its identity columns:

| Fields | Definition |
|---|---|
| `returned_target_ratio` | `norm(z) / norm(h_init)` |
| `actual_post_pre_ratio` | `norm(h_post) / norm(h_pre)` |
| `returned_full_target_norm`, `actual_pre_norm` | Recorded scalar norms of `z` and `h_pre` |
| `common_target_ratio` | `returned_full_target_norm / actual_pre_norm` |
| `relative_target_realization_error` | `norm(h_post - z) / norm(z)` |
| `target_post_cosine` | Cosine between `z` and `h_post` |
| `requested_actual_displacement_cosine` | Cosine between `z - h_pre` and `h_post - h_pre` |
| `reference_vector_relative_difference` | `norm(h_pre - h_init) / norm(h_init)` |
| `reference_norm_relative_difference` | Absolute difference between the recorded norms of `h_pre` and `h_init`, divided by the recorded norm of `h_init` |
| `pre_vector_finite`, `post_vector_finite` | Whether every coordinate of the corresponding captured vector is finite |
| `requested_displacement_zero`, `actual_displacement_zero` | Numeric 0/1 flags for zero displacement; blank if the required vector is unavailable |
| `pre_norm_reconstruction_relative_error`, `post_norm_reconstruction_relative_error` | Absolute difference between the captured-vector norm and its recorded scalar norm, divided by that recorded norm |

The reference-sensitivity table needs:

| Fields | Definition |
|---|---|
| `source_vectors_finite` | Both captured `h_pre` and `h_post` are finite |
| `common_reference_displacement_cosine` | Cosine between `z - h_pre` and `h_post - h_pre`; equals the corresponding vector-table value |
| `init_reference_displacement_cosine` | Cosine between `z - h_init` and `h_post - h_pre` |
| `paired_valid` | Both displacement cosines are defined and finite |
| `paired_signed_difference`, `paired_absolute_difference` | Initial-reference cosine minus common-reference cosine, and its absolute value |
| `init_reference_delta_norm`, `init_reference_delta_zero` | Norm of `z - h_init`, and whether that norm equals zero |

Do not replace undefined cosines with zero: a cosine is undefined when a vector is unavailable or either displacement is zero. Paired differences must be blank when `paired_valid` is false. Scalar norm ratios may still be available when captured vectors contain nonfinite values.

### RQ2: checkpoint measurements

Supply all nine checkpoints at edit counts `50,100,150,200,250,300,500,750,1000` for each configuration. `locality_percent` is the checkpoint's teacher-forced agreement with the unedited model's predictions.

Each row must include every combination of the prefixes `rewrite_` and `locality_` with these suffixes:

```text
mean_p, mean_q, mean_d, mean_kappa, mean_abs_norm_deviation,
mean_p_squared, mean_q_squared, mean_kappa_squared, mean_angle_degrees
```

For the same input, let `h0` be the unedited Base representation, `h` the edited representation, `r = norm(h0) > 0`, `u = h0/r`, and `delta = h-h0`. Per-case quantities are:

```text
p     = dot(delta, u) / r
q     = norm(delta - dot(delta, u)*u) / r
d     = norm(delta) / r
kappa = norm(h) / r
angle = arccos(dot(h, h0) / (norm(h)*norm(h0))), in degrees
```

Compute each suffix as the mean across the fixed evaluation cases for its prompt context. `mean_abs_norm_deviation` is the mean of `abs(kappa-1)`. `mean_p_squared`, `mean_q_squared`, and `mean_kappa_squared` are means of per-case squares, not squares of the means. They must retain the identity `2*mean_p + mean_p_squared + mean_q_squared = mean_kappa_squared - 1`. All required checkpoint measurements must be finite. Rewrite geometry uses subject-last representations; locality geometry uses prompt-last representations.

### RQ3: doses and paired case scores

`intervention_doses.csv` contains each endpoint state at orthogonal reduction fractions `0,0.25,0.5,0.75,1`. Each `delta_*` is the intervention score minus that state's native score in percentage points. The two indicator columns encode a positive LOC change together with either no EFF/GEN loss, or losses no greater than one percentage point. Compute indicators from unrounded changes and store numeric 0/1 values.

`same_norm_conditions.csv` contains both reduction fractions `0.25,0.5`, both operations `perp,projection_match`, and all endpoints `LOC,EFF,GEN` for every state. `scope` must be `B_common_all_families`: the same feasible request IDs are used across both operations and all three input families at a given state and dose. `delta_pp` is each operation's mean score change from the matched native baseline. `n_cases` must agree across operations and endpoints; use zero for an unavailable comparison, with its `delta_pp` left blank.

`rq3_selected_case_scores.csv` supplies individual scores for the paired bootstrap. Include states with eligible cases at both doses. The families are `locality,rewrite,rephrase`; for each dose, conditions are `native`, `perp_f025`/`perp_f050`, and `projection_match_f025`/`projection_match_f050`. Repeat the native scores for each dose's eligible case set. Within a state and dose, every family/condition must list identical case IDs in identical order. The 50% eligible set must be a subset of the 25% set; `eligible_at_both_doses` identifies their intersection.

`endpoint_value` is a fraction in `[0,1]`: Base prediction agreement for locality and complete-target token accuracy for rewrite/rephrase. For rewrite/rephrase, `n_target_tokens` includes one terminal EOS/EOT token, `semantic_accuracy` excludes that terminal token, and `terminator_accuracy` is its 0/1 correctness. The code checks

```text
endpoint_value = ((n_target_tokens-1)*semantic_accuracy + terminator_accuracy) / n_target_tokens
```

Rewrite/rephrase targets need at least one content token and one terminal token. The token-decomposition fields may be blank for locality. The analysis computes paired differences and intervals from these case scores; no expected-result tables are needed.

## Figure inputs and optional performance table

`analysis.plot_core_figures` reads `checkpoints.csv` and `rewrite_endpoints.csv` from the same input directory. The endpoint table has one row per configuration, with state fields, `row_id`, `context=rewrite`, `edit_count=1000`, and `mean_p,mean_q,mean_kappa,mean_abs_norm_deviation`. Its row IDs and geometry values must exactly match the corresponding final checkpoints' rewrite measurements.

Pass `--contrasts-file` the `rq1_rq3/same_norm_paired_contrasts.csv` generated by `analysis.reproduce` in its output directory. The plotter uses the computed paired estimates and intervals; this is not an additional externally supplied reference table.

An optional `endpoint_tf_1000.csv` may accompany the inputs. It has one row per configuration with state fields; `n_rewrite,n_rephrase,n_locality` each equal to 1,000; `activation_intervention=none`; a `target_span` description containing `EOS/EOT`; and finite `eff_tf_percent,gen_tf_percent,locality_percent` values in `[0,100]`. The analysis checks and exports supplied endpoint scores; it does not regenerate token predictions.
