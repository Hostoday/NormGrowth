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

Place your benchmark JSON files in the repository-root [data/](data/README.md) directory:

```text
data/
  zsre/zsre_3k.json
  counterfact/counterfact.json
```

Create the dataset subdirectories as needed. Dataset files are local inputs and are ignored by Git. Use `--data_path data/zsre/zsre_3k.json` for the main editing runner, or the corresponding CounterFact path. `BNG_DATA_ROOT` defaults to this `data/` directory.

The default experiment selects the first **1,000 valid requests**, preserves their input order, uses **editing batch size 1**, and sets seed 42. It requires at least 1,000 valid requests; it does not silently run a smaller experiment. Dataset order must match across methods.

At checkpoint `t`, EFF/GEN evaluate the first `t` edited requests and their rephrase prompts. Locality always uses **all locality prompts attached to the initial 1,000 selected requests**, compared with unedited Base predictions. This panel stays fixed, including requests that have not yet been edited. For example, at 50 edits EFF/GEN cover 50 requests while locality still covers the original 1,000-request panel. Add `--do_eval` to the main editing command to evaluate checkpoints `50,100,150,200,250,300,500,750,1000` and the final edit; the trajectory runner also evaluates the Base state at step 0.

Prepare the required model/tokenizer and method-specific covariance or projector assets separately. Capture and intervention scripts also require the checkpoints and evaluation protocols described in the experiment instructions. The source runners expose individual stages; they do not provide a single command for all 40 experimental conditions.

Default asset paths are derived from the repository location. Relative `BNG_*` environment settings are resolved from the repository root; relative command-line paths are resolved from the working directory. Keep datasets in `data/`, other local inputs in `inputs/`, and generated results in `outputs/` or `build/`.

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
