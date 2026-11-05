# Taylor decoder 검증 수정

## 원인과 수정 범위

`2fa4ae2`의 smoke에서 `SAE decode is not consistent with affine natural feature contributions`가
발생했다. pinned BatchTopKSAE는 `decoder(z) + b_dec`인 affine decoder다.
기존 `decode(one_hot) - decode(0)` 방식은 큰 bias를 FP32로 더했다가 빼면서 작은 decoder
coefficient를 잃을 수 있다. 이 오차가 큰 latent 값으로 증폭되는 실패를 CPU에서 재현했다.
사용자 GPU에서의 상세 오차는 아직 계측하지 않았으며, 재실행으로 수정 효과를 확인해야 한다.

수정은 Taylor 전용 코드에만 적용한다. 기존 Head-KL 점수·행동 평가·SAE·공통 validation 코드는 변경하지 않는다.

- 검증된 bias 없는 `nn.Linear`의 `decoder.weight` column을 직접 사용한다.
- 실제 column 크기를 보존하며, decoder bias를 뺀 뒤 얻는 근사나 unit normalization은 사용하지 않는다.
- public `decode(z)`와 **같은 dense FP32 연산 그래프**의 출력을 정확 비교한다.
- sparse column mapping은 FP64에서 확인한다(`rtol=atol=1e-10`).
- 서로 다른 dense/sparse FP32 합산 순서의 차이는 진단값으로 기록한다. 이를 곧바로 비선형 decoder로 판정하지 않는다.
- 기존 native BF16 head/edit parity 허용오차는 변경하지 않는다.

회귀 테스트는 큰 bias 상쇄, 잘못된 column 순서, 비선형 decoder, 지원하지 않는 decoder,
nonfinite weight/bias 및 FP32 합산 순서 차이를 포함한다. GPU 통과 여부는 별도 smoke 결과로만 판단한다.

## 이미 restore/setup/validate가 끝난 Pod

먼저 새 fix commit을 fetch하고, 안내받은 40자 commit으로 checkout한다. 실행 중인 scoring이
없어야 하며, 소스에 로컬 변경이 있으면 강제로 버리지 않는다. 데이터는 다시 받지 않는다.

```bash
cd /workspace/Event-SAE-Taylor
git status --short
git fetch origin research/taylor-comparison
```

고정 commit checkout 이후:

```bash
python -m pytest -q tests/test_head_taylor_*.py
git rev-parse HEAD > "$STEP/source-commit.txt"
set -o pipefail
python -u "$CLI" score \
  --config "$CFG" \
  --reference-result "$REF" \
  --baseline-scores "$BASE" \
  --output "$STEP/outputs/smoke-v2.json" \
  --benchmark-kl --execute \
  2>&1 | tee "$STEP/outputs/smoke-v2.log"
```

기존 공통 validation 코드는 바뀌지 않아, **같은 Pod·같은 환경·같은 config/weights**라면
이미 통과한 수치 검증을 다시 할 필요는 없다. 환경·config·weights가 바뀌면 다시 validate한다.
새 smoke는 반드시 다시 통과해야 한다. `smoke-v2.json`이 이미 있다면 덮어쓰지 말고 다른 이름을 사용한다.

추가 진단값:

```text
decoder_vector_source: declared_nn_linear_weight_columns_no_bias_subtraction
decoder_affine_check: exact_native_vs_dense_fp32_graph_and_fp64_column_equivalence
decoder_affine_checks: 실제 검사 횟수
max_decoder_fp64_column_abs_error: FP64 column mapping 차이
max_decoder_affine_abs_error: FP32 sparse/native 합산 차이(진단값, 통과 기준 아님)
```

Smoke가 통과한 뒤 기존 RunPod 안내의 full scoring과 비교 단계로 진행한다.
