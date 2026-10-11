# Event-SAE HeadRank

기존 실험을 보존하면서 **출력층 민감도로 아직 행동 평가하지 않은 SAE feature의
중요도 순위**를 예측하는 후속 개발 저장소다. `headrank.py`는 비학습 순위 실험이고,
`headpredict.py`는 별도의 **개발용 성공률 하락 보정 예측기**다. 두 결과를 혼동하지 않는다.

**현재 구현: 오프라인 점수 계산 → 예측/평가 계획 고정 → native 행동 평가 연결 →
개입 검증·중단/재개 → paired 결과 분석. 기존 v1 전체 점수 계산은 완료했지만
새 feature 행동 평가와 이번 추가 코드의 GPU 검증은 미완료다. CPU/mock 테스트는 실제 로봇·GPU 검증이 아니다. 예측 개선이나
논문 기여가 입증된 상태는 아니다.**

## 현재 실행 경로 — 기존 Full10과 같은 규모

개발 브랜치: `research/calibrated-head-tail`. 원래 `main`의 `183eaa9`와 이미 계산한
점수/행동 결과는 보존한다. 이 변경은 자동 commit/push 또는 GPU 실행을 하지 않는다.

- **점수 대상:** 저장된 6,009 active feature. v1 `full-scores.json`을 검증 후 재사용한다.
- **행동 평가 대상:** 미평가 feature **16개** + raw/identity **2조건**.
- **평가량:** 10 task × 20 초기상태 × 18조건 = **3,600 rollout**, 재시도 예산은 별도.
- **후보 선정:** eligible Tail/Mean/Event-max의 원래 Top-3 합집합을 보장하고,
  남는 자리는 사전 고정한 순위 불일치/무작위 규칙으로 채운다. 기존에 이미 평가한
  22개와 팀원에게 결과를 들은 feature는 제외한다. 따라서 전 사전의 Top-3가 아니라
  **미평가 후보 집합의 Top-3** 비교다. Direction-excess는 기본적으로 보조 비교군이다.
- **비용:** 기존 RTX4090 condition-time 단순 환산 약40.9시간. 실행 시간/성능/검정력 보장이 아니다.
- **중요한 한계:** 기존 자료에서 Tail과 Mean의 순위가 거의 같았다. 이 구현은 완주와
  공정한 비교를 지원하는 것이며, Tail 우월성이 이미 확인됐다는 뜻이 아니다.

전체 명령은 [실행 안내 G: Tail Full10 + 보정 예측기](docs/runbook.md#g-tail-full10--보정-예측기)를 따른다.
아래 기존 8개/500회 예시는 작은 검증용이다. `plan --preset full10`은 16개/3,600회용이다.

새 Pod에서 같은 모델을 재다운로드해 **수정 시각만** 달라진 경우는 G2의 `snapshot-check`로
현재 pinned 파일 내용과 수치 검증을 확인한 뒤, `prepare-run --snapshot-receipt`로 재사용한다.
점수 JSON이나 모델 파일 시각은 수정하지 않는다. GPU/라이브러리/모델 내용 차이는 계속 차단하며,
과거에 기록하지 않은 전체 모델 내용 해시까지 소급 검증했다고 주장하지 않는다.

재설계한 예측기는 먼저 기존 16개 행동 결과로 feature-held-out 개발 검증을 한다.
출력 변화 **크기**, 선택 변경률, 작업별/7생성위치별 **행동-bin 분포 반응**을 사용하며,
Event/크기 기반 모델도 같은 라벨과 회귀 규칙으로 보정해 비교한다. 최종 model을 학습했다고
바로 새 rollout이 승인되지 않는다. 예측기 개발과 Tail의 고정된 행동 평가 경로는 분리되어 있다.

## 기존 실험 보존

- 독립 저장소: [Event-SAE-HeadRank](https://github.com/jiyeon-yoon/Event-SAE-HeadRank), 기본 브랜치 `main`.
- 초기 개발은 `research/headrank` 브랜치에서 진행했다.
- 기준 commit: `0a406829f2fd184c081e3c695ab67f373857566c`.
- `--no-hardlinks` 독립 복제로 원본 파일과 `.git`을 공유하지 않는다.
- 분리 당시 기존 remote를 제거했다. 게시 대상은 새 `Event-SAE-HeadRank` 저장소이며,
  원래 `Event-SAE-Head`에는 push하지 않는다.
- 기존 결과를 읽기 전용 입력으로 사용한다. 입력과 겹치거나 이미 존재하는 새 출력은 거부한다.
- 복제된 `docs/`와 기존 CLI는 이전 실험 기록이다. 새 작업의 시작점은 이 README와 `headrank.py`다.

출처: [고정 입력과 복제 기준](research_metadata/head_rank_origin.json).
기존 [full10 공개 결과](https://huggingface.co/datasets/jiyeony/event-sae-head-full10-results/tree/69c33d6a943c253dc97cf9a3968095276f7e9540)는 새 실험 결과가 아니다.

## 새 점수와 비교 규칙

OpenVLA, Layer 31 SAE, 저장된 hidden과 원래 forward의 정확한 sparse latent를 고정한다.
동일 query의 7개 생성 위치에서 `KL(original || edited)`를 계산한다. 문맥은 고정하고
토큰을 재생성하지 않는다. 새 데이터 수집·SAE 학습·로봇 rollout은 필요하지 않지만 GPU 계산 비용은 든다.

| 점수 | 계산 | 확인하려는 점 |
|---|---|---|
| Mean Head-KL | 7개 위치 평균 → query 평균 → task 동일 가중 평균 | 기존 평균 민감도 |
| Tail Head-KL | 7개 위치 평균 → task별 상위10% query 평균 → task 평균 | 드문 큰 반응이 평균에 가려지는가 |
| Direction excess | query 실제 제거 KL − 고정 대조4개 query KL 중앙값 → task 평균 | 같은 변경 크기에서도 feature 방향이 특별한가 |
| Event-max | coverage ≥ 0.5인 task/event 행의 `matrix_raw` 최댓값 | 이벤트 관련성 비교군 |

전체 Head sample은 250 episode × 8시점 × 7위치 = 14,000 readout이다.
task당 200 query이므로 Tail은 상위20개를 사용한다. 비활성 feature의 query도 0으로 포함한다.
episode별 query 수가 같은 task 안에서 다르면 자동 추정하지 않고 중단한다.
Tail은 **저장된 probe의 꼬리**이지 전체 시계열의 희귀 이벤트나 실패확률 꼬리가 아니다.

대조 방향은 seed1729로 선택한 다른 SAE decoder 방향4개다. 자연적인 다른 feature 제거가
아니라 **anchor 제거의 크기와 활성 시점**을 빌린 방향 대조다. BF16 이전 norm을 맞추고,
cast 이후 실제 norm도 검사한다. 허용오차 상대5% + 절대1e-5를 넘는 active row가 하나라도
있으면 feature의 excess를 `null`로 표시한다. 실패 pair를 빼거나 기준을 자동 완화하지 않는다.
이 보수적 기준은 수치 검사용이며, 성공률 예측에 최적이라는 근거는 없다.

### Event 평균과 최댓값 구분

기존 full10의 `event_aligned`는 **행 평균**이었다. 이번 `event_max`는 같은 discovery250
자료의 최댓값이며 Top-3 `[7024, 24830, 2955]`를 실제 입력에서 확인했다. public500
집계본을250개 비교군인 것처럼 사용하지 않는다. episode 집합은 같아도 Head는8시점,
Event는 전체 retained 시점을 사용하므로 시점까지 같지는 않다. inherited SAE/cluster와
역사적 초기상태·generation flag 검증 한계도 남는다. 기존 결과로 Head-KL > Event-max를 주장하지 않는다.

## 먼저 실행할 것

새 저장소를 현재 폴더로 사용한다. GPU 없이 비용과 CPU 테스트를 확인할 수 있다.

```bash
python scripts/openvla/headrank.py budget --features 8 --cases-per-task 5
PYTHONPATH=. python -m pytest -q tests/test_head_rank_*.py
```

로컬 검증은 실제 개입 hook + 가짜 모델/환경의 native runner, 계획/입력 검증,
회별 journal, 중단·재개, CPU 분석을 포함한다. 아래 실행기는 아직 실제 GPU smoke를
통과하지 않았다. 기존 Dr.VLA의 scikit-learn 보정 테스트1개는 이 macOS 환경에서
native process가 종료되어 회귀 검사에서 제외했다.

검사 명령과 결과는 실행 시점의 소스에 대해 기록한다. 현재 HeadRank/HeadPredict 및
관련 기존 회귀 검사 명령은 다음과 같다. 위 Dr.VLA 환경 문제 1개는 명시적으로 제외한다.
원래 `Event-SAE-Head`의 Git 상태는 변경 없이 깨끗하다.

```bash
python -m pytest -q tests/test_head_rank*.py tests/test_head_predict*.py tests/test_output_head*.py tests/test_head_stage1.py tests/test_head_followup.py -k 'not test_generality_calibration_requires_independent_labels_and_refits_folds'
```

이후 명령은 **이전 입력을 이미 복원한 RunPod 경로 예시**다. Pod에서 새 저장소를
별도 디렉터리에 복제하고 그 디렉터리에서 실행한다. 기존 저장소에 변경을 덮어쓰지 않는다.

```bash
cd /workspace
git clone https://github.com/jiyeon-yoon/Event-SAE-HeadRank.git
cd Event-SAE-HeadRank
git rev-parse HEAD
```

실험에는 사용할 commit을 고정하고 기록한다. 실행 도중 `git pull`로 코드를 바꾸지 않는다.
필요 입력은 복원한 결과 bundle, 같은 `ae.pt`와 옆의 `config.json`, 검증용 OpenVLA snapshot이다.
이 계산에는 원본369개 dense 파일 전체를 다시 받을 필요가 없다.

```bash
STEP=/workspace/headrank-v1
RESTORED=/workspace/head-stage1-v1/restored
REF="$RESTORED/results/full10-followup-v1"
SAE=/workspace/head-stage1-v1/sae/trainer_0/ae.pt
MODEL=/workspace/head-inputs/openvla-spatial
EVENT="$RESTORED/inputs/full10/discovery_event_scores.pt"
CFG="$STEP/config/action-runtime.yaml"

python scripts/openvla/headfollowup.py stage1-setup \
  --restored "$RESTORED" --sae "$SAE" --snapshot "$MODEL" \
  --output "$STEP/config"

python scripts/openvla/output_head_sensitivity.py validate \
  --config "$CFG" --allow-model-execution

python scripts/openvla/headrank.py score \
  --config "$CFG" --reference-result "$REF" --event-scores "$EVENT" \
  --output "$STEP/smoke.json" --execute
```

원래 GPU의 `passed`를 빌리지 않고 현재 GPU에서 head/edit parity를 새로 검증한다.
smoke는 task별1 episode의8 query, 총80 query/560 readout과 기존16개 개발용 feature다.
`--execute` 없이는 계산하지 않는다. 새 rollout은0회다.

**`elapsed_seconds`, `diagnostics.num_excess_eligible_features`, `num_norm_invalid_readouts`를
먼저 확인한다. 수치/시간 확인 없이 아래 전체 계산으로 넘어가지 않는다.**
smoke의 Event-max는 전체250 기준이므로10 episode smoke와 공정한 예측 비교를 주장하지 않는다.

## 전체 점수와 아직 평가하지 않은 후보

전체 점수 계산은 cache의 활성6,009 feature를 대상으로 한다. 이것은6,009개의 로봇 평가가 아니다.
대조4개 때문에 기존 단순 Head-KL보다 계산량이 늘어난다. 비용은 smoke로 따로 추정한다.

```bash
python scripts/openvla/headrank.py score \
  --config "$CFG" --reference-result "$REF" --event-scores "$EVENT" \
  --feature-set all-active --full-readouts --max-pairs 5000000 \
  --output "$STEP/full-scores.json" --execute
```

동일 장치의 Mean Head-KL도 재계산하며 원래 값과의 차이는 diagnostic으로 기록한다.
query별 결과, 실제 norm, 대조방향 IDs, 입력·구현 hash를 저장한다.

계획은 기존 pilot/full10 합집합22개를 기본 제외한다. 팀원이 이미 본 다른 feature의
결과도 제외해야 한다. 아래 예시를 복사해 추가 `feature_ids`를 작성하고, 실제 팀 확인 뒤
`reviewed_for_team_prior_outcomes`를 `true`로 바꾼다. 빈 리스트는 자동 승인되지 않는다.

```bash
mkdir -p configs/local
cp configs/research/openvla/headrank_known_features.example.json \
  configs/local/headrank-known.json

python scripts/openvla/headrank.py plan \
  --scores "$STEP/full-scores.json" --registry configs/local/headrank-known.json \
  --features 8 --cases-per-task 5 --output "$STEP/panel-draft.json"
```

기본8개는 지표별 순위 후보4개 + 순위 불일치2개 + seeded random2개를 중복 없이 뽑는다.
각 방법의 원래 Top-3 포함 여부를 기록하며 누락 feature를 낮은 순위로 대체하지 않는다.
모든 지표의 점수가 finite여야 네 지표 패널을 고정한다. excess `null`이 있으면 **중단**하므로
현재 구현 그대로 유료 검증을 진행할 수 없다. Mean/Tail 진단은 남기고 비교 규칙을 먼저 검토한다.

평가 manifest가 없으면 `needs_eval_manifest`다. `cases`는 원래 split의 evaluation
부분에서만 task당5개 초기상태를 고정 seed로 선택한다. 성공 결과를 읽지 않는다.
기존 discovery/validation 상태를 평가에 섞거나, 과거 상태 검증/라벨 사용 이력을
새로운 것으로 바꾸지 않는다. 기존20개/task 모두 쓰려면 **cases와 plan 모두**20으로 바꾼다.

```bash
SPLIT="$RESTORED/inputs/full10/split.json"

python scripts/openvla/headrank.py cases \
  --split "$SPLIT" --cases-per-task 5 --output "$STEP/eval-cases.json"

python scripts/openvla/headrank.py plan \
  --scores "$STEP/full-scores.json" --registry configs/local/headrank-known.json \
  --eval-manifest "$STEP/eval-cases.json" \
  --features 8 --cases-per-task 5 --output "$STEP/panel-frozen.json"

python scripts/openvla/headrank.py verify-plan --plan "$STEP/panel-frozen.json"
```

hash는 예측·선정 규칙·case·구현을 연결하지만 사람이 결과를 보지 않았음을 증명하지 않는다.
같은 기존 task/초기상태를 사용하므로 **새 feature의 전향적 평가이지, 보지 않은 task나
새로운 데이터 수집에 대한 일반화 검증이 아니다.**

## 행동 평가 연결과 안전 검증

새 HeadRank 실행기는 다음을 연결한다. 기존 `output_head_sensitivity.py run`에 새 panel을
전달하는 방식이 아니며, 이전 결과는 절대로 resume 원본으로 사용하지 않는다.

- `panel_feature_ids` → 같은 OpenVLA 최종 Layer 31의 원래 all-row SAE 개입 hook.
- raw(개입 없음), identity(패널 첫 feature, α=1), 각 feature 완전 제거(α=0).
- case의 task/원본 trial/초기상태 SHA256 → 실제 LIBERO 초기상태와 사전 대조.
- 모든 조건에 같은 `SHA256(base_seed:task_id:trial)` 기반 episode seed 적용.
- 모델·processor·SAE를 한 번 로드해 재사용하되 환경과 hook은 회마다 생성/해제.
- 먼저 case별 raw→identity를 실행하고 **행동 trace와 성공 결과의 완전 일치**를 검사.
  모든 control case가 통과해야 feature 제거를 시작한다. 실제 α=0 hook이 해당 feature를
  모든 policy step에서 제거했는지 기록을 검사한다. 자연적으로 비활성인 feature는
  허용하되 활성 수0을 그대로 기록한다.
- 예외/개입 오류/증거 누락은 작업 실패(False)가 아니라 **실험 중단**이다.
- 매 회 시작 기록을 먼저 저장하고 결과·행동·개입 증거를 hash로 묶는다. 재개할 때
  정상 완료된 case는 건너뛰고, 손상된 기록이나 승인되지 않은 재시도는 거부한다.

`prepare-run`은 새 forward/rollout 없이 계획과 현재 장치 parity를 연결한다.
모델 가중치 전체 content hash도 읽으므로 모델 크기만큼 디스크 I/O 시간이 든다.
**코드를 검토한 뒤 clean commit으로 고정해야 준비/유료 실행이 가능하다.
실험 CLI는 자동 commit, 다운로드, 승인 없는 유료 실행을 하지 않는다.**
코드를 고정한 버전에서 parity와 점수를 연결한다. 지원되는 예외는 출처를 정확히 고정한
`183eaa9` v1 점수의 검증된 재사용뿐이다. 원래 scorer/수치 의존성과 새 planner/runtime
출처를 별도로 보존하며, 원래 점수의 hash나 implementation을 고쳐 재서명하지 않는다.
임의로 변경된 scorer/환경/입력은 그대로 차단한다.

```bash
python scripts/openvla/headrank.py prepare-run \
  --config "$CFG" --plan "$STEP/panel-frozen.json" \
  --scores "$STEP/full-scores.json" --registry configs/local/headrank-known.json \
  --eval-manifest "$STEP/eval-cases.json" --split "$SPLIT" \
  --reference-result "$REF" --max-rollouts 500 \
  --output "$STEP/execution.json" --run-output "$STEP/runs"

python scripts/openvla/headrank.py run --experiment "$STEP/execution.json"
```

마지막 명령은 **실행하지 않고** 조건·예산·승인 hash를 보여준다. 기본8 feature ×
10 task ×5 case는 control 포함500회이며, 재시도 여유는0이다. 재시도 여유를 원하면
**준비 시점**에 예를 들어 `--retry-budget 2 --max-rollouts 502`로 명시한다.
실행 중 예산을 임의로 늘리거나 hash를 고쳐 우회하지 않는다.

내용을 확인하고 실제 GPU 실행을 승인할 때만:

```bash
EXEC_HASH=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["execution_hash"])' "$STEP/execution.json")
CODE_REV=$(git rev-parse HEAD)

python scripts/openvla/headrank.py run \
  --experiment "$STEP/execution.json" --execute \
  --approved-execution-hash "$EXEC_HASH" --expected-code-revision "$CODE_REV" \
  --max-new-attempts 2
```

이것은 첫 case의 **raw1회 + identity1회**를 실행하고 회 경계에서 멈춘다. 실제 제거
개입 자체는 아직 실행하지 않았으며, 기존 numerical edit parity와 CPU hook 검사가
그 전에 요구된다. `status: paused`는 오류가 아니다. 첫 pair가 일치했어도 전체
control 검사가 끝난 것은 아니다. 장시간 실행은 tmux 안에서 아래처럼 재개한다.

```bash
python scripts/openvla/headrank.py run \
  --experiment "$STEP/execution.json" --execute \
  --approved-execution-hash "$EXEC_HASH" --expected-code-revision "$CODE_REV"

python scripts/openvla/headrank.py status --experiment "$STEP/execution.json"
```

이미 완료된 case는 재실행하지 않는다. 중간에 **미완료/실패 attempt**가 남으면 자동
재시도하지 않는다. 사전 승인한 reserve가 있을 때만 같은 run 명령에
`--retry-incomplete`를 추가한다. 중단된 attempt도 예산을 사용한 것으로 센다.
두 프로세스가 같은 journal을 동시에 실행하지 못하게 잠근다.
완료된 journal의 상태 확인과 분석은 GPU가 없어도 가능하다.

## 완료 후 분석과 보존

```bash
python scripts/openvla/headrank.py analyze \
  --experiment "$STEP/execution.json" --output "$STEP/analysis-v1"
```

모든 예정 case가 완료되고 raw/identity가 일치해야 `analysis.json`, `report.md`를 쓴다.
성공률 변화는 동일 초기상태의 `raw_success − ablation_success` 평균이다. 음수는 개선이며
0으로 자르지 않는다. 고정 패널 내 Spearman, 주 지표와 Mean Head-KL/Event-max의 차이,
원래 Top-K가 모두 평가된 경우에만 Top-K 평균 하락을 계산한다. Bootstrap은 task 내에서
같은 case를 모든 feature/방법에 공통으로2000회 재표집한다(seed1729, 95% pointwise CI).
작은 패널의 개선이나 통계적 검정력을 보장하지 않으며, undefined 값은0으로 바꾸지 않는다.

`execution.json`, 전체 `runs/`(hook 증거 포함), 분석 결과와 정확한 소스 snapshot을 함께
보존한다. 다운로드해 다른 위치에 놓았으면 `status`/`analyze`에 `--run-root 복원한/runs`
를 주면 된다. 과거 원본 입력의 절대 경로나 GPU 없이 저장 결과를 검증·재분석할 수 있다.
이는 simulator 재실행이나 GPU parity 재검증과는 다르다. GPU 재실행은 원래 승인된
경로·코드·모델·장치·입력 파일과 새 parity 조건을 충족해야 한다.

## 행동 평가 예상 시간

기존 RTX4090 실측3,600회/40.877 condition-hour를 단순 비례했다.
계산은 `(새 feature 수 + raw/identity 2조건) × 10 task × task당 초기상태 수`다.

| 규모 | feature | task당 초기상태 | 총 rollout | 추정시간 |
|---|---:|---:|---:|---:|
| 작은검증 | 8 | 5 | 500 | 5.7시간 |
| 같은8개를 기존밀도로 | 8 | 20 | 2,000 | 22.7시간 |
| 기존실험과 같은규모 | 16 | 20 | 3,600 | 40.9시간 |
| 더넓은feature검증 | 24 | 20 | 5,200 | 59.0시간 |

점수 계산·다운로드·설정·검증·재시도는 별도다. 여러 지표가 고른 **서로 다른 feature 합집합**만
평가하므로 지표 수를 다시 곱하지 않는다. pilot 결과를 보고 방법을 수정하면 그 feature는
개발용이 되어 다른 새 feature의 검증 비용이 추가된다. 이 예산은 논문용 검정력 보장이 아니다.

## 남은 실제 검증과 연구 판단

1. v1 GPU 점수 계산은 이미 완료했다. 이번 추가 descriptor 경로를 사용할 때만 그 경로의 실제 GPU 검증이 필요하다.
2. 팀원의 기존 평가 ID를 확인한 뒤 규칙·새 feature 후보·paired case를 확정한다.
3. 구현한 native 실행기의 실제 GPU control pair smoke와 모든 control 검증을 수행한다.
4. 새 feature를 각각 제거하고, 실제 hook 증거와 paired 성공률 하락을 확인한다.
5. 완료 결과를 CPU 분석한다. Full10 preset의 주 지표는 Tail이며 Excess/보정 예측은
   보조 분석이다. 분석 후 잘 나온 방법으로 주 지표를 바꾸지 않는다.

새 feature에서 Mean Head-KL과 Event-max보다 실제 예측이 개선되어야 후속 기여를 주장할 수 있다.
출력 변화가 크다는 것만으로 유해성·의미 있는 행동 개념·실패를 입증하지 않는다.
전체32,768 feature 예측, 새 모델/task 일반화, 로봇 정책 성공률 개선도 현재 주장 범위가 아니다.

## License

MIT. 원본 및 외부 모델·데이터의 저작권과 라이선스를 유지한다.
