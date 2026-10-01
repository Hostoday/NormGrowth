# 공개 수치 자료와 CPU 재현

현재 원고의 관측값과 참조 표는 [manuscript/](manuscript/README.md)에 있다. [snapshot 명세](analysis_snapshot.json)가 기본 분석에서 사용하는 파일과 표본 범위를 지정한다. 모델 가중치, 데이터셋의 prompt·answer, 원시 hidden tensor는 포함하지 않는다. CSV의 수치 문자열은 보존하고 서버 경로 열만 제거했으며, 원자료와 공개 사본의 SHA256은 [출처 기록](provenance/data_sources.json)에 있다.

| 분석 | 현재 입력과 범위 |
|---|---|
| RQ1 | 40,000개 edit의 scalar 및 vector 측정; finite 39,473개, paired cosine 39,257개 |
| RQ2 주 분석 | 40 trajectory × 9 checkpoint = 360개 |
| RQ2 민감도 | geometry 제한 337개; 붕괴한 여섯 trajectory 제외 306개 |
| RQ3 | 직교 축소 40개 상태, 동일 norm 대조 34개 상태, 직접 대비 204개 |
| 최종 성능 | [endpoint_tf_1000.csv](manuscript/endpoint_tf_1000.csv)의 40개 endpoint |

저장소 루트에서 실행한다.

```bash
python -m analysis.reproduce --output-dir build/results
python -m analysis.plot_core_figures --output-dir build/figures
```

## 평가와 통계

EFF_TF·GEN_TF는 정답 prefix를 조건으로 첫 답 token부터 EOS/EOT까지의 argmax 정확도를 계산한 값이다. 사례별 token 평균을 먼저 구한 뒤 사례 간 평균한다. LOC는 동일한 locality 정답 prefix에서 편집 모델과 Base의 예측 일치율이며 종료 token을 추가하지 않는다. 1,000-edit 성능표와 RQ3의 editing 지표는 모두 이 TF 정의를 따른다.

RQ2는 전체 360개 상태를 주 분석으로 사용한다. 두 문맥의 평균 q·κ·D가 모두 3 이하인 337개 상태 및 Llama–MEMIT Native·SPHERE·SADR 여섯 trajectory를 제외한 306개 상태는 별도의 민감도 분석이다. 부분순위상관은 norm 편차, 편집 횟수, trajectory를 구분해 통제하며 성분 구성의 추가 연관에는 총 변위를 통제한다. 5,000회 bootstrap은 trajectory의 아홉 checkpoint를 함께 재표집한다.

RQ3는 각 입력의 원 prompt-last에서 개입한다. 같은 상태·감소율에서 두 조작과 세 입력 유형의 공통 적격 사례를 대응시키며, 10,000회 paired bootstrap으로 상태별 구간을 구한다. 전체 평균은 적격 상태마다 동일 가중치를 부여한다. 구간은 다중비교 보정 전이며 전체 평균의 CI는 추정하지 않는다.

## 자료 구분

`manuscript/expected/`는 독립적으로 저장된 원고 참조 표다. 재현 코드는 관측값에서 다시 계산한 결과와 이 표를 비교한다. 최종 TF 성능표는 저장 점수의 범위·표본·출처를 검증해 내보내며, 원시 예측이나 모델 추론을 새로 생성하지 않는다.

`manuscript/` 밖의 기존 CSV는 이전 snapshot과 공통 집계 참조를 보존한 자료다. 특히 `performance_1k_40.csv`는 과거 자유 생성 성능표이며 현재 원고의 TF 성능표로 사용하지 않는다. 기본 분석의 파일 선택은 snapshot 명세를 따른다.
