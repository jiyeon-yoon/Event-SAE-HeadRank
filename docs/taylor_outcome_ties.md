# Taylor 비교 보고서의 outcome 동률 처리 수정

기존 `comparison-v1`에서 과제별 성공률을 먼저 float로 계산하고 평균내면
수학적으로 같은 성공률이 서로 다른 float가 될 수 있었다. 실제 새16개 집단의 `31089`는
harm17 / repair17인데 약1.39e-18의 순손상으로 계산되어 다른 zero-effect feature와
Spearman 동률이 깨졌다.

수정: 각 과제의 `(harm-repair)/n`, `(harm+repair)/n`을 정수 기반 Fraction으로 유지하고,
동일 과제 가중 평균 후 한 번만 float로 변환한다. 과제별 case 수가 다른 경우도 동일 가중을
유지한다. 예측 점수에 반올림·임의 epsilon을 적용하지 않는다.

- Taylor/Head-KL/Event-max 점수 파일·모델·SAE·행동 결과는 바뀌지 않는다.
- 새 rollout·GPU scoring·validate는 필요 없다.
- 기존 comparison-v1은 보존하고 비교 명령의 출력 경로만 `comparison-v2`로 바꿔 실행한다.
- 새 비교 보고서에는 outcome_arithmetic 및 comparison_implementation_sha256가 기록된다.
- 이 수정 전의 새16개 순손상 Spearman은 최종 수치로 인용하지 않는다.

```bash
python "$CLI" compare \
  --scores "$STEP/outputs/full-scores.json" \
  --panel "original_full10=$REF/analysis/paired_effects.json" \
  --panel "new_headrank=$STEP/new-panel/analysis.json" \
  --output "$STEP/outputs/comparison-v2"

sed -n '1,220p' "$STEP/outputs/comparison-v2/report.md"
```
