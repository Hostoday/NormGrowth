# 공개 수치 자료와 CPU 재현

이 폴더는 RQ1–RQ3의 저장된 관측을 재집계하기 위한 수치 자료다. 모델 가중치, 원시 hidden vector, prompt·target 문장, 학습 데이터셋은 포함하지 않는다. CSV의 수치·식별자 셀은 원본 문자열을 보존했고 서버 경로 열은 제거했다. 외부 원자료와 공개 사본의 SHA256 및 열 선택 내역은 [출처 기록](provenance/data_sources.json)에 있다.

| 파일 | 단위·크기 | 용도 |
|---|---|---|
| [rq1_per_edit_40000.csv](rq1_per_edit_40000.csv) | 40조건 × 1,000개 편집 | target/post-update ratio 중앙값, OLS, 비팽창 비율 |
| [rq2_checkpoints_360.csv](rq2_checkpoints_360.csv) | 360개 서로 다른 checkpoint | 공통 제외 기준 적용 및 두 prompt 문맥의 상관 |
| [rq3_orthogonal_states_doses_200.csv](rq3_orthogonal_states_doses_200.csv) | 40상태 × 5강도 | 직교 축소의 상태별 성능 변화와 동일 가중 평균 |
| [rq3_same_norm_conditions_480.csv](rq3_same_norm_conditions_480.csv) | 40상태 × 2강도 × 2조작 × 3 family | 동일 최종 norm에서 직교·평행 조작 비교 |
| [performance_1k_40.csv](performance_1k_40.csv) | 40개 1,000-edit 조건 | 완성된 1k 자유 생성 성능표, ENCORE 보완 포함 |
| [figure4_rewrite_endpoints_34.csv](figure4_rewrite_endpoints_34.csv) | 34개 endpoint | Figure 4의 rewrite 성분 분해 |
| [figure5_prompt_overlay_2022.csv](figure5_prompt_overlay_2022.csv) | 337 checkpoint × 2 prompt × 3 지표 | Figure 5의 정확한 좌표·표식 자료 |
| `expected/` | 기존 결과의 참조 집계 6개 | 재집계 수치·제외 집합과 원래 보고값 비교 |

저장소 루트에서 다음 명령을 실행한다.

```bash
python -m analysis.reproduce --output-dir build/reproduced
python -m analysis.plot_core_figures --output-dir build/figures
```

첫 명령은 입력 SHA256을 검사하고 RQ1·RQ2·RQ3를 다시 계산해 기존 참조 표와 비교한다. `build/reproduced/validation.json`에 실제 검사 결과가 저장된다. 두 번째 명령은 기존 STIX 글꼴·색상·표식·축 범위를 사용하여 Figure 4·5의 PNG·PDF·SVG를 만든다. Figure 5의 모든 좌표를 360개 원표에 공통 제외 기준을 다시 적용한 값과 대조한다. 다른 논문 그림은 문서 패키지의 완성된 파일을 사용한다.

## 측정과 집계의 의미

- **RQ1:** `returned_target_ratio`는 target 초기 norm으로 나눈 반환 target norm이다. `actual_post_pre_ratio`는 같은 편집의 실제 update 전후 표현 norm ratio다. 두 측정의 문맥·분모 차이를 유지하며 전달 메커니즘의 직접 측정으로 해석하지 않는다. 비팽창 횟수와 π는 원 분석과 동일하게 `ratio <= 1 + 1e-6`을 사용한다. π의 분모는 조건별 모든 1,000개 편집이다.
- **RQ2:** locality는 prompt-last, rewrite는 subject-last의 Base-relative 표현이다. Llama는 H9, GPT-2 XL은 H18이다. 각각 고정 1,000사례의 기하량 평균을 사용하고 y축은 해당 checkpoint의 고정 locality 1,000사례 출력 일치율이다. 두 문맥의 q·κ·D 중 하나라도 3을 초과하면 checkpoint 전체를 공통 제외한다. 360개 중 337개(zsRE 186, CounterFact 151)가 남는다. 상관은 checkpoint 평균들의 기술통계이며 checkpoint는 독립 replicate가 아니다. `partial_spearman_given_D`는 각 문맥의 **평균 절대 norm 이탈 D**를 통제한 rank residual 간 Pearson 상관이다. 전체 변위 길이를 통제한 값이 아니다.
- **RQ3 직교 축소:** 두 모델·두 데이터셋의 40상태를 동일 가중 평균한다. 강도는 0·25·50·75·100%이며 0은 같은 편집 checkpoint의 무개입 기준이다. 성능 차이는 percentage points다. 원래 저장된 상태별 분류 지시자는 수치적 영점 처리까지 보존한다.
- **RQ3 동일 norm 대조:** `perp`는 직교 성분 축소, `projection_match`는 직교 축소로 정한 최종 norm을 맞추는 평행 투영 조절이다. 25·50%는 **기준 직교 축소량**이며 평행 성분을 그 비율로 줄인다는 뜻이 아니다. 각 상태·강도에서 세 family 모두 가능한 사례의 공통 교집합만 사용한다. 40상태 중 34상태에서 정의되며, 6개 Llama MEMIT Native·SPHERE·SADR 상태는 `n_cases=0`, 결과 `NaN`으로 남긴다. 누락값을 0점으로 바꾸지 않는다. 강도별 총 state-case 수는 3,391과 3,309이며, 최종 평균은 사례 수로 가중하지 않은 **상태별 동일 가중 평균**이다.
- **평가 차이:** 1k 성능표의 EFF·GEN은 자유 생성이다. RQ3의 EFF_TF·GEN_TF는 complete-target teacher-forced 지표다. 각 family의 원래 prompt-last 표현에 개별 직접 개입하므로 LOC도 locality 입력 자체를 조절한 결과다. 두 성분을 동시에 조절한 실험은 포함하지 않는다.

이 CPU 재현은 저장된 측정의 집계와 그림을 확인한다. 모델 학습, raw hidden 추출, 개입 forward, 사례별 bootstrap을 다시 수행하지 않는다. 포함된 CI는 원래 계산된 값이며, 이 명령에서 다시 생성한 CI가 아니다.
