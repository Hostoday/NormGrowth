# Local datasets

Put the dataset JSON files you will use for model editing in this directory:

```text
data/
  zsre/zsre_3k.json
  counterfact/counterfact.json
```

Create these subdirectories as needed. Only this README is tracked by Git; dataset files and generated measurements remain local. No dataset or saved experiment result is distributed with the code.

Run commands from the repository root. The main editing runner accepts `--data_path data/zsre/zsre_3k.json` or `--data_path data/counterfact/counterfact.json`. Other runners use the hyphenated `--data-path` option where shown in their help. The default dataset root is this directory; set `BNG_DATA_ROOT` to use another location.

The input may be a JSON list or an object with a `data` list. The main runner accepts normalized EasyEdit requests, zsRE-style `src`/`alt`/`subject`/`rephrase`/`loc`/`loc_ans` records, and CounterFact records containing `requested_rewrite`, `paraphrase_prompts`, and `neighborhood_prompts`. Each selected request must include a valid rewrite target and its locality prompt/answer pairs. Do not replace locality prompts with the rewrite prompt or sample them from a separate dataset.

The defaults select the first **1,000 valid requests**, in file order, with editing batch size **1** and seed **42**. Supply at least 1,000 valid requests. Keep the input file and request order identical across methods.

The evaluation cohorts are:

| Checkpoint | EFF/GEN | Locality |
|---|---|---|
| 50 edits | First 50 edited requests and their rephrase prompts | All locality prompts attached to the selected 1,000 requests |
| 500 edits | First 500 edited requests and their rephrase prompts | The same fixed locality panel |
| 1,000 edits | All 1,000 edited requests and their rephrase prompts | The same fixed locality panel |

Locality compares predictions with the unedited Base model. If a request contains multiple locality prompt/answer pairs, keep all of them; the number of locality prompts may therefore exceed 1,000. The main runner writes the selected requests to its output `requests.json` before editing. Use that file for later evaluation instead of selecting another cohort.

See the [experiment instructions](../experiments/README.md) for model setup and evaluation commands. Statistical analysis reads measurement tables generated from these experiments, as described in the [analysis input schema](../analysis/INPUTS.md).
