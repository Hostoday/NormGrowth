# 모델 편집·측정·개입 코드

이 폴더에는 MEMIT·AlphaEdit 편집, hidden-state 측정, 직교 개입과 동일 norm 평행 대조를 위한 실행기와 내부 의존성을 담았다. 실험 설정과 코드를 제공하며, 측정 결과와 생성된 파일은 포함하지 않는다.

논문의 EFF·GEN은 complete-target teacher-forced 점수이며 평가 target에 포함된 EOS/EOT도 유지한다. LOC는 동일한 teacher-forced 문맥에서 편집 전후 모델의 토큰 예측 일치율이다. 자유 생성 평가 경로도 소스에 포함되어 있으므로 실행 시 사용할 평가 규약을 구분해야 한다.

모델·tokenizer, 데이터셋, covariance·projection cache는 외부에서 준비해야 한다. 측정·개입 실행기에는 편집 checkpoint와 단계별 protocol·capture도 필요하다. 아래 명령은 각 단계의 진입점이며, 전체 40조건을 한 번에 실행하는 통합 파이프라인은 제공하지 않는다.

## 포함한 코드

| 단계 | 진입점·구현 | 범위 |
|---|---|---|
| MEMIT·AlphaEdit 편집 | [run_rgr_batch.py](scripts/run_rgr_batch.py), [MEMIT](EasyEdit/easyeditor/models/memit/memit_main.py), [AlphaEdit](EasyEdit/easyeditor/models/alphaedit/AlphaEdit_main.py) | Native·NAS·ENCORE·SPHERE·SADR의 opt-in 구현과 target 최적화·outer update |
| 순차 편집과 artifact 기록 | [analyze_edit_count_gain_trajectory.py](diagnostics/analyze_edit_count_gain_trajectory.py) | 원 프로젝트의 순차 실행기. HN을 포함한 과거 보조 옵션도 소스에 존재하며 논문 모든 조건의 기본 설정은 아님 |
| Llama locality geometry | [capture_scalar_checkpoint_h9.py](diagnostics/capture_scalar_checkpoint_h9.py) | Base + cumulative delta 복원, locality prompt-last H9 |
| Llama rewrite geometry | [capture_rewrite_backfill.py](model_code/capture_rewrite_backfill.py) | inventory 기반 고정 평가 집합, rewrite subject-last H9 |
| GPT-2 XL checkpoint 평가 | [gpt2_checkpoint_analysis.py](diagnostics/gpt2_checkpoint_analysis.py) | 모델·run·data 경로를 받는 H18 capture와 checkpoint 평가 |
| 직교 개입, Llama–zsRE | [run_llama_intervention_extension.py](diagnostics/run_llama_intervention_extension.py) | 기존 protocol v2의 모델 실행과 집계 |
| 직교 개입, GPT-2 XL 두 데이터셋 | [run_gpt2_intervention_extension.py](diagnostics/run_gpt2_intervention_extension.py) | `--dataset zsRE` 또는 `CounterFact` |
| 직교 개입, Llama–CounterFact | [run_llama_counterfact_intervention.py](diagnostics/run_llama_counterfact_intervention.py) | 10조건과 보존된 source manifest |
| 동일 norm 평행 대조, Llama–CounterFact | [run_llama_counterfact_parallel_control.py](diagnostics/run_llama_counterfact_parallel_control.py) | 기하적으로 가능한 case의 총 Base-axis projection 조절 |
| 동일 norm 평행 대조, 나머지 세 조합 | [run_all_cohort_parallel_control.py](diagnostics/run_all_cohort_parallel_control.py) | `llama_zsre`, `gpt2_zsre`, `gpt2_counterfact` |
| 평가 규약 | [eval_cumulative_generation_locality.py](evaluate/eval_cumulative_generation_locality.py), [target_text_contract.py](diagnostics/target_text_contract.py) | 자유 생성 성능과 TF locality의 구분, 실제 EOS/EOT 보존 |

AlphaEdit의 SPHERE 분기는 각 층의 update를 투영한 결과를 다음 층의 key·residual 계산에 반영한다. `Linear`는 저장 방향을 유지하고 GPT-2 `Conv1D`는 weight와 update를 모두 `[output,input]`으로 전치한 뒤 정규화·투영하고 원 저장 방향으로 복원한다. 따라서 두 backbone 모두 MLP 입력 방향에 같은 제약을 적용한다.

## 환경과 모델 없는 확인

현재 EasyEdit 환경의 실제 패키지 버전을 [저장소 requirements.txt](../requirements.txt)에 기록했다. 공개 환경 이름은 `normgrowth`이며 Python 3.9.7, PyTorch 2.1.0(CUDA 12.1), Transformers 4.46.2를 사용한다. [environment.yml](../environment.yml)로 환경을 생성한다.

```bash
conda env create -f environment.yml
conda activate normgrowth
python -B experiments/scripts/run_rgr_batch.py --help
python -B experiments/diagnostics/gpt2_checkpoint_analysis.py --help
python -B experiments/diagnostics/run_all_cohort_parallel_control.py --help
python -B experiments/model_code/capture_rewrite_backfill.py --help
python -B experiments/model_code/smoke.py
```

`--help`는 모델을 로드하지 않는다. `smoke.py`는 평행 대조의 직교 성분 보존·목표 norm과 불가능한 입력의 처리를 확인하고 소스 문법을 검사한다.

## 경로 설정과 외부 자산

저장소 루트에서 위 명령을 실행한다. 실행기 파일을 다른 작업 디렉터리에서 지정해도 기본 경로는 저장소 위치를 따른다. 경로 기본값은 [model_code/paths.py](model_code/paths.py)에서 설정한다.

| 환경변수 | 의미 | 기본 경로 |
|---|---|---|
| `BNG_RESEARCH_ROOT` | 과거 연구 artifact 트리의 루트 | `inputs/research/` |
| `BNG_OUTPUT_ROOT` | 원 실행기가 요구하는 `outputs` 트리 | `$BNG_RESEARCH_ROOT/Residual_gain_regulization/outputs/` |
| `BNG_DATA_ROOT` | 원 데이터셋 경로의 기본 루트 | `inputs/datasets/` |
| `BNG_MODEL_ROOT` | 로컬 모델 폴더의 루트 | `models/` |
| `BNG_LLAMA_MODEL` | Llama 모델과 tokenizer가 있는 폴더 | `$BNG_MODEL_ROOT/Meta-Llama-3-8B-Instruct/` |
| `BNG_CACHE_ROOT` | 보조 모델 cache 루트 | `build/cache/` |
| `BNG_SCRATCH_ROOT` | EasyEdit trainer 임시 파일 | `build/scratch/` |
| `BNG_BLIP2_ASSET_ROOT` | 보조 BLIP2 구성 파일의 루트 | `inputs/blip2/` |

```bash
export BNG_RESEARCH_ROOT=inputs/research
export BNG_OUTPUT_ROOT=inputs/research/Residual_gain_regulization/outputs
export BNG_DATA_ROOT=inputs/datasets
export BNG_MODEL_ROOT=models
export BNG_LLAMA_MODEL=models/Meta-Llama-3-8B-Instruct
```

환경변수의 상대경로는 저장소 루트 기준으로 해석한다. 외부 디스크의 경로를 사용자가 직접 지정할 수도 있다. 위 예시는 논문 자료를 제공하지 않으며 실제 자산은 해당 위치에 준비해야 한다. CLI에서 명시적으로 넘기는 상대경로는 일반 명령행 규약대로 호출 작업 디렉터리를 기준으로 한다. BLIP2는 현재 논문의 실험 대상이 아니며, 보조 코드에서 요구하는 구성 파일도 포함하지 않았다.

GPT-2 XL의 일반 평가기는 `--model-path`, `--run-dir`, `--data-path`, `--output-root`를 받는다. 개입 실행기는 protocol/source manifest에서 checkpoint·request·model 경로를 읽는다. 환경변수는 코드의 기본 경로를 바꾸며, manifest 내부의 경로·SHA256·case 순서는 자동 변환하지 않는다. 자산을 옮기면 해당 protocol의 경로도 실제 위치와 일치시켜야 한다.

평행 대조 실행기에는 먼저 생성한 직교 개입 출력과 평가 protocol, native hidden·prediction이 필요하다. `--audit-only`와 `--aggregate-only`에도 이 입력을 준비해야 한다. 일부 source-audit는 protocol 파일을 생성한다.

실제 실행에는 다음 자료가 필요하다.

- Llama-3-8B-Instruct 또는 GPT-2 XL의 동일한 모델·tokenizer revision.
- 정규화된 1,000개 request, canonical case 순서와 fit500/eval100 분할.
- 각 editor·방법·데이터셋의 cumulative parameter delta, sidecar metadata, run configuration.
- 편집을 다시 할 경우 covariance 통계, AlphaEdit projector와 NAS anchor 등 방법별 자산.
- 개입을 계속할 경우 기존 protocol·source manifest·native capture·prediction과 corrected 평가 기록.

RQ2의 주 rewrite 측정은 subject-last이고 locality는 prompt-last다. RQ3에서는 rewrite·rephrase·locality 각각의 원 prompt-last에 직접 개입한다. 따라서 RQ3을 rewrite subject-last 상관의 동일 위치 인과 검증으로 해석하지 않는다. 현재 논문에 보고하는 대조는 직교 성분의 부분 축소와 동일 최종 norm의 평행 projection 조절이다. 원 실행기에는 과거 random 방향 등 추가 조건이 남아 있지만, 공개된 현재 Results에 이 조건들을 추가로 포함한다는 뜻은 아니다.

## 설정과 라이선스

ENCORE의 8조건 설정은 [조건별 설정표와 실행 안내](hparams/README.md), `hparams/{MEMIT,AlphaEdit}/*_encore.yaml`에 제공하며 AlphaEdit의 `L2=10`을 포함한다. 일반 `llama3-8b.yaml`은 기본 템플릿이므로 논문의 조건별 설정 대신 사용하지 않는다.

EasyEdit 코드는 [원 MIT License](EasyEdit/LICENSE)를 보존했다. AlphaEdit 상류 소스의 [MIT License](model_code/licenses/AlphaEdit-LICENSE)도 함께 제공한다. 모델 가중치와 데이터셋은 각 배포처의 별도 이용 조건을 따른다.
