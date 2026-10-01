# 모델 편집·측정·개입 코드

이 폴더에는 현재 연구 프로젝트의 MEMIT·AlphaEdit 수정본과, RQ2·RQ3에 사용한 실행기의 소스 및 내부 의존성을 담았다. 저장된 공개 수치의 CPU 재집계와 모델을 다시 실행하는 작업은 요구 자료가 다르다. CPU 분석 안내는 저장소 루트 README를 따른다.

모델 가중치, 데이터셋의 prompt·answer, covariance·projection cache, 편집 checkpoint, 원시 hidden tensor는 포함하지 않았다. 따라서 이 저장소만으로 논문의 전체 40조건을 새로 학습하고 동일 수치를 재생성하는 통합 실행을 제공한다고 볼 수 없다. 아래 실행기는 구현 확인과 외부 자산을 준비한 모델 실험의 출발점이다.

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

AlphaEdit의 SPHERE 분기는 **각 층의 update를 투영한 뒤 그 결과를 다음 층의 key·residual 계산에 반영한다**. 현재 SPHERE 구현은 실제 모듈 타입을 확인한다. `Linear`는 저장 방향을 유지하고 GPT-2 `Conv1D`는 weight와 update를 모두 `[output,input]`으로 전치한 뒤 행 정규화·부분공간 계산·투영을 수행하고 원 저장 방향으로 복원한다. 두 backbone 모두 MLP 입력 방향을 제어하며 적용 방향·차원·버전은 SPHERE 로그에 기록한다. 저장된 수치의 실험 버전은 `data/analysis_snapshot.json`의 `source_bundle`과 출처 기록으로 식별한다. [소스 manifest](../data/provenance/code_sources.json)는 원본과 공개본 SHA256 및 경로 수정 내역을 제공한다. 각 구현 파일의 원본 및 공개본 hash를 확인할 수 있다.

## 환경과 모델 없는 확인

기존 검증 환경은 Python 3.9, PyTorch 2.1.0, Transformers 4.46.2다. EasyEdit의 package import가 모델 외의 dataset·trainer 모듈도 불러오므로 모델 소스 환경은 CPU 수치 분석 환경보다 의존성이 많다. [requirements.txt](requirements.txt)는 이 환경에서 사용한 직접 의존성 목록이다. 새로운 환경에서 전체 dependency를 다시 설치한 검증은 수행하지 않았다.

```bash
python -m pip install -r experiments/requirements.txt
python -B experiments/scripts/run_rgr_batch.py --help
python -B experiments/diagnostics/gpt2_checkpoint_analysis.py --help
python -B experiments/diagnostics/run_all_cohort_parallel_control.py --help
python -B experiments/model_code/capture_rewrite_backfill.py --help
python -B experiments/model_code/smoke.py
```

`--help`는 모델을 로드하지 않는다. `smoke.py`는 평행 대조가 가능한 입력에서 직교 성분을 보존하고 목표 norm을 맞추는지, 불가능한 입력을 임의로 보정하지 않는지 확인하며 소스 문법도 검사한다. 모델 추론·학습 성능 검증과는 다르다. 실행 범위는 [smoke 기록](../data/provenance/model_code_smoke.json)에 남긴다.

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

GPT-2 XL의 일반 평가기는 `--model-path`, `--run-dir`, `--data-path`, `--output-root`를 받는다. 개입 실행기는 보존된 protocol/source manifest에서 checkpoint·request·model 경로를 읽는다. 환경변수는 코드의 기본 경로를 바꾸며, **manifest 안에 저장된 경로·SHA256·case 순서는 자동으로 바꾸지 않는다.** 기존 protocol을 다른 파일의 것으로 조용히 덮어쓰면 checkpoint 동일성 검사가 성립하지 않는다. 자산을 이동할 때는 원래의 경로 해석을 유지하거나, 별도 실행 폴더에 변경된 경로와 provenance를 기록하고 source audit를 다시 수행해야 한다.

평행 대조 실행기는 원래의 직교 개입 출력과 corrected 평가 protocol, native hidden·prediction의 동일성을 먼저 확인한다. 해당 자료 없이 `--audit-only`나 `--aggregate-only`를 실행하는 것도 불가능하다. 일부 source-audit는 protocol 파일을 생성·검증하므로 읽기 전용 검사 명령으로 취급하지 않는다. 논문의 CPU 집계에는 포함된 수치 데이터용 분석 진입점을 사용한다.

실제 실행에는 다음 자료가 필요하다.

- Llama-3-8B-Instruct 또는 GPT-2 XL의 동일한 모델·tokenizer revision.
- 정규화된 1,000개 request, canonical case 순서와 fit500/eval100 분할.
- 각 editor·방법·데이터셋의 cumulative parameter delta, sidecar metadata, run configuration.
- 편집을 다시 할 경우 covariance 통계, AlphaEdit projector와 NAS anchor 등 방법별 자산.
- 개입을 계속할 경우 기존 protocol·source manifest·native capture·prediction과 corrected 평가 기록.

RQ2의 주 rewrite 측정은 subject-last이고 locality는 prompt-last다. RQ3에서는 rewrite·rephrase·locality 각각의 원 prompt-last에 직접 개입한다. 따라서 RQ3을 rewrite subject-last 상관의 동일 위치 인과 검증으로 해석하지 않는다. 현재 논문에 보고하는 대조는 직교 성분의 부분 축소와 동일 최종 norm의 평행 projection 조절이다. 원 실행기에는 과거 random 방향 등 추가 조건이 남아 있지만, 공개된 현재 Results에 이 조건들을 추가로 포함한다는 뜻은 아니다.

## 설정과 라이선스

ENCORE의 실제 8조건 설정은 [조건별 설정표와 실행 안내](hparams/README.md), `hparams/{MEMIT,AlphaEdit}/*_encore.yaml`에 제공한다. 이 preset은 저장된 실행값을 보존하며 AlphaEdit의 `L2=10`을 포함한다. 일반 `llama3-8b.yaml`은 기존 기본 템플릿이므로 논문 설정 대신 사용하지 않는다. ENCORE의 Llama 세 조건은 2026-09-24 교정 재실험의 λ/MPES 설정으로 동기화했다. 8개 preset이 전체 40조건의 통합 실행을 제공하는 것은 아니다.

EasyEdit 코드는 [원 MIT License](EasyEdit/LICENSE)를 보존했다. AlphaEdit 상류 소스의 [MIT License](model_code/licenses/AlphaEdit-LICENSE)도 함께 제공한다. 모델 가중치와 데이터셋은 각 배포처의 별도 이용 조건을 따른다.
