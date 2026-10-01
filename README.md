# Beyond Norm Growth

**Base-Relative Geometry of Hidden-State Drift in Sequential Knowledge Editing**

Experimental and analysis code for MEMIT and AlphaEdit with Native, NAS, ENCORE, SPHERE, and SADR on Llama-3-8B-Instruct and GPT-2 XL, using zsRE and CounterFact.

This repository contains source code, configuration files, and execution instructions. Experiment measurements, result tables, figures, model weights, and datasets are generated or obtained separately.

## Environment

Create the `normgrowth` environment from the recorded package versions:

```bash
conda env create -f environment.yml
conda activate normgrowth
```

[requirements.txt](requirements.txt) records the package versions from the existing EasyEdit environment. [environment.yml](environment.yml) sets the public environment name and Python version. GPU execution also requires a compatible NVIDIA driver. For statistics and plotting alone, install [analysis/requirements.txt](analysis/requirements.txt) in a separate Python 3.9 environment.

## Model experiments

Start with the [experiment instructions](experiments/README.md) for environment setup, editing and evaluation entry points, geometry capture, and interventions. The [hyperparameter guide](experiments/hparams/README.md) describes the supplied ENCORE configurations.

Prepare the required model/tokenizer, benchmark inputs, and method-specific covariance or projector assets. Capture and intervention scripts also require the checkpoints and evaluation protocols described in the experiment instructions. The source runners expose individual stages; they do not provide a single command for all 40 experimental conditions.

Default asset paths are derived from the repository location. Relative `BNG_*` environment settings are resolved from the repository root; relative command-line paths are resolved from the working directory. Keep local inputs in `inputs/` and generated results in `outputs/` or `build/`; these directories are ignored by Git.

## Statistics and figures

In the `normgrowth` environment, run from the repository root after collecting your measurements into the [analysis input format](analysis/INPUTS.md):

```bash
python -m analysis.reproduce --data-dir inputs/measurements --output-dir build/results
python -m analysis.plot_core_figures --data-dir inputs/measurements --contrasts-file build/results/rq1_rq3/same_norm_paired_contrasts.csv --output-dir build/figures
```

The analysis computes RQ1 norm and vector comparisons, RQ2 geometry–locality associations and sensitivity analyses, and RQ3 paired contrasts and bootstrap intervals from the supplied measurements. It does not run model inference. No saved paper result or reference table is required. See [analysis instructions](analysis/README.md) for individual modules and statistical scope.

## Checks

```bash
python -m analysis.check_release
python experiments/model_code/smoke.py
```

The source check verifies Python syntax, local documentation links, portable paths, and exclusion of experiment data. The component smoke check runs in the model environment and verifies the norm-matched intervention construction without model weights. CI runs the source check and analysis command-line checks with the smaller analysis environment; numerical analyses require your measurement inputs.

## License and citation

Project code is provided under the [MIT license](LICENSE). Vendored EasyEdit and AlphaEdit sources retain their licenses and notices; see [source attribution](experiments/NOTICE.md). Citation metadata is in [CITATION.cff](CITATION.cff).
