# ENCORE experiment configurations

The eight `*_encore.yaml` files contain the condition-specific ENCORE settings used in the paper. The two generic `llama3-8b.yaml` files are templates and do not replace these presets.

| Model | Dataset | Editor | Additional norm λ | MPES qualifying observations | Preset |
|---|---|---|---:|---:|---|
| Llama-3-8B-Instruct | zsRE | MEMIT | 10 | 4 | [YAML](MEMIT/llama3-8b_zsre_encore.yaml) |
| Llama-3-8B-Instruct | CounterFact | MEMIT | 20 | 2 | [YAML](MEMIT/llama3-8b_counterfact_encore.yaml) |
| GPT-2 XL | zsRE | MEMIT | 40 | 5 | [YAML](MEMIT/gpt2-xl_zsre_encore.yaml) |
| GPT-2 XL | CounterFact | MEMIT | 10 | 4 | [YAML](MEMIT/gpt2-xl_counterfact_encore.yaml) |
| Llama-3-8B-Instruct | zsRE | AlphaEdit | 0 | 4 | [YAML](AlphaEdit/llama3-8b_zsre_encore.yaml) |
| Llama-3-8B-Instruct | CounterFact | AlphaEdit | 0 | 1 | [YAML](AlphaEdit/llama3-8b_counterfact_encore.yaml) |
| GPT-2 XL | zsRE | AlphaEdit | 0 | 2 | [YAML](AlphaEdit/gpt2-xl_zsre_encore.yaml) |
| GPT-2 XL | CounterFact | AlphaEdit | 0 | 2 | [YAML](AlphaEdit/gpt2-xl_counterfact_encore.yaml) |

`encore_mpes_top1_steps` counts the accumulated observations at which the target satisfies the top-1 criterion. It does not count consecutive successes or total gradient updates. All configurations exclude the first rewrite context from the MPES check. For AlphaEdit, additional norm λ=0 adds MPES while retaining the base AlphaEdit `L2=10` setting.

Llama uses at most 25 target-optimization steps, learning rate 0.1, and loss layer 31. GPT-2 XL uses at most 20 steps, learning rate 0.5, and loss layer 47. Both models use displacement-penalty coefficient 0.5, KL coefficient 0.0625, and relative delta clamp 0.75. Covariance statistics use 100,000 samples; the MEMIT covariance coefficient is 15,000 for Llama and 20,000 for GPT-2 XL.

## Relation to the ENCORE settings

The [ENCORE v2 Appendix G](https://arxiv.org/html/2502.01636v2#A7), referenced by the experiment scripts, reports settings that vary across model, dataset, and editor combinations. The implementation maps the paper's cutoff `+n` to `n+1` accumulated qualifying observations. This describes the cutoff mapping rather than equivalence of the entire optimization procedure.

- GPT-2 XL MEMIT uses λ40/cutoff+4 from zsRE Table 11 and λ10/cutoff+3 from CounterFact Table 8, corresponding to five and four qualifying observations. GPT-2 XL AlphaEdit uses two observations, corresponding to cutoff+1 in Tables 10 and 6.
- Llama CounterFact MEMIT uses λ20 and two observations, corresponding to cutoff+1 in Table 8.
- Llama zsRE MEMIT uses λ10 and four observations.
- Llama AlphaEdit uses four observations for zsRE and one for CounterFact. Both use additional norm λ=0.

## Running a configuration

Place the datasets under the repository-root `data/` directory and prepare the model and method-specific statistics. Run from the repository root. For GPT-2 XL–zsRE–MEMIT:

```bash
python experiments/scripts/run_rgr_batch.py \
  --editing_method MEMIT \
  --hparams_path experiments/hparams/MEMIT/gpt2-xl_zsre_encore.yaml \
  --data_path data/zsre/zsre_3k.json \
  --output_dir outputs/gpt2_zsre_memit_encore \
  --batch_size 1 --sample_size 1000 --selection prefix --seed 42 \
  --append_eos_to_target 1 --save_model 0 --do_eval
```

The runner defaults to the first 1,000 valid normalized requests, editing batch size 1, prefix selection, and seed 42. With `--do_eval`, it scores checkpoints at `50,100,150,200,250,300,500,750,1000` edits and the final edit, using evaluation batch size 1. EFF and GEN evaluate all requests edited so far; locality always uses all locality entries belonging to the initial selected cohort, with Base predictions recorded before editing. Without `--do_eval`, the runner performs editing only. For CounterFact, use `data/counterfact/counterfact.json` and the matching preset. Change the editor, YAML, input dataset, and output directory together when selecting another configuration.

The direct editing runner preserves YAML values unless an ENCORE CLI override is supplied. `diagnostics/analyze_edit_count_gain_trajectory.py` applies its own ENCORE CLI defaults, so use the direct runner above to apply these presets without additional overrides. See the [experiment guide](../README.md) for checkpoint measurement and intervention inputs.
