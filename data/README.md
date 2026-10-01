# 공개 수치 자료와 CPU 재현

이 폴더는 snapshot 명세의 실험 버전에 해당하는 canonical 순서의 RQ1–RQ3 관측을 제공한다. 추가 편집 순서 `o0_seed20260905`의 SPHERE·SADR trajectory는 제외한다. 모델 가중치, 원시 hidden vector, prompt·target 문장, 학습 데이터셋은 포함하지 않는다. 수치 CSV 셀은 원본 문자열을 보존하고 서버 경로 열은 제거한다. 기존 locality `mean_d`는 `locality_mean_d`로 통일한다. 원자료·공개 사본의 SHA256 및 선택 열은 [출처 기록](provenance/data_sources.json)에 있다.

기본 설계는 40조건 × 9 checkpoint = 360개다. 실제 가용·유지 관측 수, checkpoint 목록과 누락 trajectory는 [snapshot 명세](analysis_snapshot.json)에 기록한다. 이 명세를 재현 코드와 그림 코드가 함께 사용하며, 부족한 관측을 보간하거나 다른 순서의 관측으로 채우지 않는다.

| 파일 | 단위 | 용도 |
|---|---|---|
| [rq1_per_edit_40000.csv](rq1_per_edit_40000.csv) | 40조건 × 1,000편집 | target/post-update ratio 중앙값, OLS, 비팽창 비율 |
| [rq2_checkpoints.csv](rq2_checkpoints.csv) | canonical checkpoint | 공통 제외 기준, 두 문맥의 q·d·κ·D 및 second moment |
| [rq3_orthogonal_states_doses_200.csv](rq3_orthogonal_states_doses_200.csv) | 40상태 × 5강도 | 직교 축소의 상태별 성능 변화와 동일 가중 평균 |
| [rq3_same_norm_conditions_480.csv](rq3_same_norm_conditions_480.csv) | 40상태 × 2강도 × 2조작 × 3 family | 동일 최종 norm에서 직교·평행 조작 비교 |
| [performance_1k_40.csv](performance_1k_40.csv) | 40개 endpoint | 1,000-edit 자유 생성 성능 |
| [figure4_rewrite_endpoints.csv](figure4_rewrite_endpoints.csv) | rewrite endpoint | Figure 4와 endpoint norm 분해 |
| [locality_endpoints.csv](locality_endpoints.csv) | locality endpoint | endpoint norm 분해 |
| [figure5_prompt_overlay.csv](figure5_prompt_overlay.csv) | 유지 checkpoint × 2문맥 × 3지표 | Figure 5의 정확한 좌표 |
| `expected/` | 독립 집계된 참조 표 8개 | 재집계 수치와 원래 보고값 비교 |

저장소 루트에서 다음 명령을 실행한다.

```bash
python -m analysis.reproduce --output-dir build/reproduced
python -m analysis.plot_core_figures --output-dir build/figures
```

첫 명령은 SHA256을 검사하고 RQ1·RQ2·RQ3, q–d 상관, endpoint norm 분해를 재집계한다. `build/reproduced/validation.json`은 실제 비교 결과와 표본 수를 기록한다. checkpoint별 에너지 성분도 `rq2_checkpoint_energy.csv`로 출력한다. 두 번째 명령은 Figure 4·5의 PNG·PDF·SVG를 만들고 모든 그림 좌표를 입력 관측과 비교한다. Figure 5와 q–d 상관은 동일한 공통 제외 집합을 사용한다.

## 측정과 집계

- **RQ1:** `returned_target_ratio`는 target 초기 norm으로 나눈 반환 target norm이다. `actual_post_pre_ratio`는 같은 편집의 update 전후 표현 norm ratio다. 두 측정의 문맥·분모 차이를 유지하며 전달 메커니즘의 직접 측정으로 해석하지 않는다. 비팽창은 `ratio <= 1 + 1e-6`, π의 분모는 각 조건의 모든 1,000개 편집이다.
- **RQ2:** locality는 prompt-last, rewrite는 subject-last의 Base-relative 표현이다. Llama는 H9, GPT-2 XL은 H18이다. 각 checkpoint에서 고정 1,000사례 기하량 평균과 고정 locality 1,000사례 출력 일치율을 연결한다. 어느 문맥이든 평균 q·κ·D 중 하나가 3을 넘으면 checkpoint 전체를 공통 제외한다. `partial_spearman_given_D`는 평균 절대 norm 이탈 D를 통제한 rank residual의 Pearson 상관이다. 전체 변위 d를 통제한 값이 아니다. checkpoint는 독립 replicate가 아니며 상관은 기술통계다.
- **Norm 분해:** `mean_p_squared`, `mean_q_squared`, `mean_kappa_squared`는 개별 사례의 제곱을 평균한 값이다. `P=2 mean_p+mean_p_squared`, `Q=mean_q_squared`, `c=-P/Q`이며 `P+Q=mean_kappa_squared-1`을 검증한다. 직교 에너지 비율은 `mean_q_squared/(mean_p_squared+mean_q_squared)`이다. 평균 p·q를 먼저 제곱하지 않는다.
- **RQ3 직교 축소:** 40상태를 동일 가중 평균한다. 강도는 0·25·50·75·100%이며 0은 같은 편집 checkpoint의 무개입 기준이다. 성능 변화는 percentage points다. 상태별 분류 지시자는 원래의 수치적 영점 처리를 보존한다.
- **RQ3 동일 norm 대조:** `perp`는 직교 성분 축소, `projection_match`는 직교 축소로 정한 최종 norm을 맞추는 평행 투영 조절이다. 25·50%는 기준 직교 축소량이다. 상태·강도별 세 family의 공통 가능 사례만 비교한다. 유효한 상태 수와 state-case 합은 강도마다 실제 `n_cases>0`인 공통 교집합에서 집계하며 snapshot·검증 보고서에 기록한다. 공통 적격 사례가 없는 상태는 `n_cases=0`, 결과 `NaN`으로 보존하고 평균은 유효 상태마다 동일한 가중치를 사용한다. 차이 행은 원정밀도 평균의 차이를 구한 뒤 표시할 때 반올림한다.
- **평가 차이:** 1k 성능표의 EFF·GEN은 자유 생성, RQ3의 EFF_TF·GEN_TF는 complete-target teacher-forced 지표다. 각 family의 원래 prompt-last 표현에 개별 개입하므로 LOC도 locality 입력 자체를 조절한 결과다.

CPU 재현은 저장된 측정의 집계와 그림을 확인한다. raw hidden 추출, 개입 forward, 사례별 bootstrap을 다시 수행하지 않는다. 포함된 CI는 원래 계산된 값이다.
