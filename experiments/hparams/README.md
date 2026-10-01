# ENCORE: 실행에 사용한 조건별 설정

`*_encore.yaml` 8개는 보고된 ENCORE 실행의 YAML과 저장된 effective 설정을 대조해 만든 실행 preset이다. 기존 `llama3-8b.yaml` 두 파일은 일반 템플릿이며 이 preset을 대신하지 않는다. 2026-09-24 ENCORE 정렬 재실험에서 교체한 Llama 세 조건을 반영한다. 파일별 근거와 hash는 [소스 기록](../../data/provenance/code_sources.json)에 있다.

| 모델 | 데이터셋 | Editor | 추가 norm λ | MPES 충족 관측 수 | Preset |
|---|---|---|---:|---:|---|
| Llama-3-8B-Instruct | zsRE | MEMIT | 10 | 4 | [YAML](MEMIT/llama3-8b_zsre_encore.yaml) |
| Llama-3-8B-Instruct | CounterFact | MEMIT | 20 | 2 | [YAML](MEMIT/llama3-8b_counterfact_encore.yaml) |
| GPT-2 XL | zsRE | MEMIT | 40 | 5 | [YAML](MEMIT/gpt2-xl_zsre_encore.yaml) |
| GPT-2 XL | CounterFact | MEMIT | 10 | 4 | [YAML](MEMIT/gpt2-xl_counterfact_encore.yaml) |
| Llama-3-8B-Instruct | zsRE | AlphaEdit | 0 | 4 | [YAML](AlphaEdit/llama3-8b_zsre_encore.yaml) |
| Llama-3-8B-Instruct | CounterFact | AlphaEdit | 0 | 1 | [YAML](AlphaEdit/llama3-8b_counterfact_encore.yaml) |
| GPT-2 XL | zsRE | AlphaEdit | 0 | 2 | [YAML](AlphaEdit/gpt2-xl_zsre_encore.yaml) |
| GPT-2 XL | CounterFact | AlphaEdit | 0 | 2 | [YAML](AlphaEdit/gpt2-xl_counterfact_encore.yaml) |

`encore_mpes_top1_steps`는 target이 top-1 조건을 충족한 누적 관측 횟수다. 연속 성공 횟수나 전체 gradient update 횟수가 아니다. 모든 조건은 첫 rewrite context를 MPES 판정에서 제외한다. AlphaEdit의 추가 norm λ=0은 MPES만 추가한다는 뜻이며, 기본 AlphaEdit의 `L2=10`은 유지된다.

공통 target 설정은 Llama 최대 25회/LR 0.1/loss layer 31, GPT-2 XL 최대 20회/LR 0.5/loss layer 47이다. 두 모델 모두 displacement-penalty coefficient 0.5, KL 0.0625, relative delta clamp 0.75를 사용했다. Covariance 표본은 100,000개이며 MEMIT covariance coefficient는 Llama 15,000, GPT-2 XL 20,000이다.

## 원 ENCORE 논문과의 관계

실행 스크립트가 참조한 [ENCORE v2 Appendix G](https://arxiv.org/html/2502.01636v2#A7)는 모델·데이터셋·조합별로 다른 hyperparameter를 보고한다. 따라서 모든 조건에 동일한 숫자를 사용하는 것이 원 논문의 규칙은 아니다. 해당 문헌의 cutoff `+n`과 로컬 코드의 누적 충족 관측 수를 구분해야 한다. 기존 실행기는 `+n`을 `n+1` 충족 관측으로 매핑했다. 이는 설정 대응이며 전체 최적화 절차가 같다는 보장은 아니다.

- GPT-2 XL MEMIT은 zsRE Table 11의 λ40/cutoff+4, CounterFact Table 8의 λ10/cutoff+3을 사용해 각각 5회·4회로 설정했다. GPT AlphaEdit는 Tables 10/6의 cutoff+1에 대응하는 2회다.
- Llama CounterFact MEMIT은 Table 8의 λ20/cutoff+1과 대응한다.
- Llama zsRE MEMIT은 교정 재실험의 λ10/4회를 사용한다.
- Llama AlphaEdit는 교정 재실험의 zsRE 4회, CounterFact 1회를 사용한다. 두 조건의 추가 norm λ는 0이다.

세 조건의 RQ1·RQ2·RQ3 및 성능표는 같은 교정된 편집 실행을 사용한다. 데이터 snapshot의 실제 trajectory 포함 범위와 누락 여부는 [분석 자료 안내](../../data/README.md)에 기록한다. 추가 편집 순서는 canonical 분석에서 제외한다.

## 실행

모델 및 canonical 데이터·통계 자산을 준비한 뒤 저장소 루트에서 실행한다. 아래는 GPT-2 XL–zsRE–MEMIT 예시다. 다른 조건은 editor, 해당 YAML, 입력 데이터와 출력 폴더를 함께 바꾼다.

```bash
python experiments/scripts/run_rgr_batch.py \
  --editing_method MEMIT \
  --hparams_path experiments/hparams/MEMIT/gpt2-xl_zsre_encore.yaml \
  --data_path inputs/datasets/zsre.json \
  --output_dir outputs/gpt2_zsre_memit_encore \
  --batch_size 1 --sample_size 1000 --selection prefix --seed 42 \
  --append_eos_to_target 1 --save_model 0
```

이 실행기는 명시적인 ENCORE CLI override가 없으면 YAML 값을 유지한다. `diagnostics/analyze_edit_count_gain_trajectory.py`는 자체 CLI 기본값으로 ENCORE 값을 덮어쓰므로 위 preset을 적용할 때는 위 직접 실행기를 사용한다. 데이터 예시 경로 자체는 canonical 1,000개 요청·순서·모델 revision의 동일성을 보장하지 않으며, 전체 평가·개입 재현에는 [모델 실험 안내](../README.md)의 외부 자료가 필요하다.
