# Model editing, measurement, and interventions

This directory contains MEMIT and AlphaEdit editing code, hidden-state measurements, orthogonal interventions, and parallel controls with matched final norms. Experiment settings and source code are provided; datasets, model weights, and generated results are supplied or produced locally.

The paper evaluates EFF and GEN with complete-target teacher forcing, retaining EOS/EOT tokens when present in the evaluation target. LOC measures agreement between the Base and edited models' token predictions under the same teacher-forced context. The source also contains free-generation evaluation; select the evaluation protocol corresponding to the reported metric.

## Environment

The package versions from the EasyEdit experiment environment are recorded in the repository [requirements.txt](../requirements.txt). The public environment is named `normgrowth` and uses Python 3.9.7, PyTorch 2.1.0 with CUDA 12.1, and Transformers 4.46.2. Run these commands from the repository root:

```bash
conda env create -f environment.yml
conda activate normgrowth
python -B experiments/scripts/run_rgr_batch.py --help
python -B experiments/diagnostics/gpt2_checkpoint_analysis.py --help
python -B experiments/diagnostics/run_all_cohort_parallel_control.py --help
python -B experiments/model_code/capture_rewrite_backfill.py --help
python -B experiments/model_code/smoke.py
```

The help commands do not load a model. `smoke.py` checks source syntax and verifies that the parallel control preserves the orthogonal component, reaches the target norm when feasible, and handles infeasible inputs.

## Dataset placement and default protocol

Create a `data/` directory at the repository root and place the datasets there:

```text
data/
├── zsre/
│   └── zsre_3k.json
└── counterfact/
    └── counterfact.json
```

These files are local inputs and are not distributed with the code. Use the corresponding path with `--data_path` in the editing runner, or `--data-path` in the trajectory runner. See the [data instructions](../data/README.md) for input preparation.

The default editing protocol selects the first **1,000 valid, normalized requests** in their original order (`prefix` selection), uses seed **42**, and applies edits sequentially with **batch size 1**. Prefix selection does not randomly resample the dataset.

At each checkpoint after `t` edits, EFF and GEN evaluate the rewrite and rephrase prompts of **all requests edited so far**, namely the first `t` selected requests. Locality uses a **fixed panel containing every locality entry associated with the initial 1,000 selected requests**, at every checkpoint. It is not independently sampled or restricted to the requests edited so far. For example, at 50 edits EFF and GEN cover the first 50 edit requests while locality covers the locality entries of all 1,000 selected requests. At 1,000 edits, both use the full selected cohort. Requests can have multiple locality entries, so the number of locality prompts can differ from 1,000.

The trajectory runner defaults to 1,000 edits and evaluates at edit counts `0,50,100,150,200,250,300,500,750,1000`, where 0 denotes the Base model. Evaluation batch sizes control forward-pass throughput and are separate from the editing batch size.

In `run_rgr_batch.py`, enable checkpoint scoring with `--do_eval`. It records the full fixed locality panel's Base predictions before editing and evaluates after `50,100,150,200,250,300,500,750,1000` edits, plus the final edit when it falls outside that schedule. The default evaluation batch size is 1. Results are written to `evaluations/step_XXXX.json` under the run output directory; `summary.json` includes `final_cumulative_evaluation`. Without `--do_eval`, the runner performs editing only.

For example, run the GPT-2 XL–zsRE–MEMIT ENCORE configuration with:

```bash
python experiments/scripts/run_rgr_batch.py \
  --editing_method MEMIT \
  --hparams_path experiments/hparams/MEMIT/gpt2-xl_zsre_encore.yaml \
  --data_path data/zsre/zsre_3k.json \
  --output_dir outputs/gpt2_zsre_memit_encore \
  --batch_size 1 --sample_size 1000 --selection prefix --seed 42 \
  --append_eos_to_target 1 --save_model 0 --do_eval
```

For CounterFact, use `data/counterfact/counterfact.json` and the matching configuration. The eight ENCORE presets and their condition-specific values are described in the [hyperparameter guide](hparams/README.md).

## Experiment entry points

| Stage | Entry point or implementation | Scope |
|---|---|---|
| MEMIT and AlphaEdit editing | [run_rgr_batch.py](scripts/run_rgr_batch.py), [MEMIT](EasyEdit/easyeditor/models/memit/memit_main.py), [AlphaEdit](EasyEdit/easyeditor/models/alphaedit/AlphaEdit_main.py) | Native, NAS, ENCORE, SPHERE, and SADR options; target optimization and weight updates |
| Sequential editing and artifact capture | [analyze_edit_count_gain_trajectory.py](diagnostics/analyze_edit_count_gain_trajectory.py) | Sequential execution, checkpoint evaluation, and optional diagnostics; historical options such as HN remain available |
| Llama locality geometry | [capture_scalar_checkpoint_h9.py](diagnostics/capture_scalar_checkpoint_h9.py) | Restore Base plus cumulative parameter deltas and capture locality prompt-last H9 states |
| Llama rewrite geometry | [capture_rewrite_backfill.py](model_code/capture_rewrite_backfill.py) | Fixed evaluation set specified by an inventory; rewrite subject-last H9 states |
| GPT-2 XL checkpoint evaluation | [gpt2_checkpoint_analysis.py](diagnostics/gpt2_checkpoint_analysis.py) | H18 capture and checkpoint evaluation with model, run, and dataset paths |
| Orthogonal intervention, Llama–zsRE | [run_llama_intervention_extension.py](diagnostics/run_llama_intervention_extension.py) | Model execution and aggregation from protocol v2 |
| Orthogonal intervention, GPT-2 XL | [run_gpt2_intervention_extension.py](diagnostics/run_gpt2_intervention_extension.py) | Select `--dataset zsRE` or `CounterFact` |
| Orthogonal intervention, Llama–CounterFact | [run_llama_counterfact_intervention.py](diagnostics/run_llama_counterfact_intervention.py) | Ten configurations specified by a source manifest |
| Matched-norm parallel control, Llama–CounterFact | [run_llama_counterfact_parallel_control.py](diagnostics/run_llama_counterfact_parallel_control.py) | Adjust the total Base-axis projection for geometrically feasible cases |
| Matched-norm parallel control, other cohorts | [run_all_cohort_parallel_control.py](diagnostics/run_all_cohort_parallel_control.py) | `llama_zsre`, `gpt2_zsre`, and `gpt2_counterfact` |
| Evaluation conventions | [eval_cumulative_generation_locality.py](evaluate/eval_cumulative_generation_locality.py), [target_text_contract.py](diagnostics/target_text_contract.py) | Free-generation evaluation, teacher-forced locality, and preservation of target EOS/EOT tokens |

These are entry points for individual stages. Running all 40 configurations requires selecting each model, dataset, editor, and method and passing the outputs to the subsequent measurement and intervention stages.

In AlphaEdit, SPHERE projects each layer's update before computing the next layer's keys and residuals. `Linear` weights retain their stored orientation. GPT-2 `Conv1D` weights and updates are both transposed to `[output, input]` for normalization and projection, then restored to their stored orientation. Both backbones therefore apply the constraint along the MLP input dimension.

## Paths and required assets

Default paths are defined in [model_code/paths.py](model_code/paths.py) relative to the repository location, including when a runner is invoked from another working directory.

| Environment variable | Purpose | Default path |
|---|---|---|
| `BNG_RESEARCH_ROOT` | Root of the research artifact tree used by existing protocols | `inputs/research/` |
| `BNG_OUTPUT_ROOT` | Output tree used by the original runners | `$BNG_RESEARCH_ROOT/Residual_gain_regulization/outputs/` |
| `BNG_DATA_ROOT` | Dataset root | `data/` |
| `BNG_MODEL_ROOT` | Local model root | `models/` |
| `BNG_LLAMA_MODEL` | Llama model and tokenizer directory | `$BNG_MODEL_ROOT/Meta-Llama-3-8B-Instruct/` |
| `BNG_CACHE_ROOT` | Auxiliary model cache root | `build/cache/` |
| `BNG_SCRATCH_ROOT` | Temporary EasyEdit trainer files | `build/scratch/` |
| `BNG_BLIP2_ASSET_ROOT` | Auxiliary BLIP2 configuration root | `inputs/blip2/` |

For example:

```bash
export BNG_RESEARCH_ROOT=inputs/research
export BNG_OUTPUT_ROOT=inputs/research/Residual_gain_regulization/outputs
export BNG_DATA_ROOT=data
export BNG_MODEL_ROOT=models
export BNG_LLAMA_MODEL=models/Meta-Llama-3-8B-Instruct
```

Relative paths in these environment variables are resolved against the repository root. Explicit relative CLI paths are resolved against the current working directory. Absolute paths to external storage are also accepted. Place the relevant assets at the configured locations. BLIP2 is auxiliary code and is not used in the paper's experiments.

The GPT-2 XL checkpoint evaluator accepts `--model-path`, `--run-dir`, `--data-path`, and `--output-root`. Intervention runners read checkpoint, request, and model paths from protocol or source manifests. When moving assets, update the manifest paths to match their new locations; environment variables do not rewrite manifest contents, hashes, or case order.

Prepare the following assets for the relevant stage:

- The model and tokenizer revisions used for Llama-3-8B-Instruct or GPT-2 XL.
- The normalized requests in their original order; measurement and intervention protocols also specify their case order and fit500/eval100 split.
- Covariance statistics and method-specific assets such as AlphaEdit projectors and NAS anchors when rerunning edits.
- Cumulative parameter deltas, metadata, and run configurations produced by editing, for checkpoint measurements.
- Protocols, source manifests, native hidden states, and predictions produced by earlier stages, for intervention and parallel-control runs.

Parallel-control runners require the orthogonal-intervention output, evaluation protocol, and native captures and predictions. Their `--audit-only` and `--aggregate-only` modes also require these inputs. Some source-audit modes write protocol files.

RQ2 measures rewrite geometry at the subject-last position and locality geometry at the prompt-last position. RQ3 intervenes at the original prompt-last position for rewrite, rephrase, and locality prompts. The reported comparisons use partial orthogonal reduction and parallel projection adjustment to the same final norm. Additional historical intervention options remain in the source.

## Configurations and licenses

The eight ENCORE configurations are supplied in `hparams/{MEMIT,AlphaEdit}/*_encore.yaml`, including AlphaEdit's base `L2=10`; see the [configuration table](hparams/README.md). The generic `llama3-8b.yaml` files are templates rather than condition-specific paper presets.

The EasyEdit [MIT License](EasyEdit/LICENSE) and the upstream AlphaEdit [MIT License](model_code/licenses/AlphaEdit-LICENSE) are preserved. Model weights and datasets remain subject to their respective providers' terms.
