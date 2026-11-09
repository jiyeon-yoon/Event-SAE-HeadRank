# Exp3 — 팀 Baseline 기준 500개·Top-5 재평가

## 1. 이번에 하는 것

| 항목 | 고정 조건 |
|---|---|
| 방법 | Mean Head-KL, Tail Head-KL, Event-max |
| 후보 선정 | 기존 LIBERO-Spatial 500개 episode, 전체 32,768개 feature에 점수 부여 |
| Head 표본 | episode마다 균등 간격 8개 query × 행동 출력 7위치 = 28,000 readout |
| Event-max | 같은 500개에서 계산한 공개 Event 점수의 canonical cluster별 최댓값 |
| 평가 후보 | 각 방법 Top-5의 합집합. 기존에 평가한 22개도 제외하지 않음 |
| 행동 평가 | 10과제 × 원래 초기 상태 50개. 각 feature를 하나씩 제거 |
| 대조 | raw, identity(α=1), 각 feature 제거(α=0) |
| seed·실행 순서 | 조건 시작 시 global seed=0 한 번 → 모델 로드 → task마다 환경 생성·env.seed(0) → trial 0~49 순서 |
| 금지 | episode별 reseeding, 완료 episode만 건너뛰기, 기존 raw/개입 결과 섞기 |

Top-5 효과는 **다섯 feature를 각각 제거한 효과의 평균**이다. 다섯 개를 동시에 제거하지 않는다.
Mean과 Tail의 Top-5 **집합**이 같으면 평균 효과도 같고, 순서가 다르면 순위 분석에서만 차이가 날 수 있다.
Top-5가 같을지 여부는 이번 500개 점수를 계산한 다음 확정한다.

- **Mean:** 같은 생성 prefix에서 feature 제거 전후의 전체 vocabulary KL을 계산한다.
  행동 출력 7위치를 평균하고, task마다 query 평균을 낸 뒤 10개 task를 동일 가중 평균한다.
- **Tail:** 같은 query별 KL 중 task별 상위 10%(개수 올림)를 평균한 다음 task를 동일 가중 평균한다.
  feature가 비활성인 query도 0으로 포함한다.
- **Event-max:** episode coverage 0.5 이상인 기존 canonical event cluster 점수의 최댓값이다.
  이벤트별 개입이나 이벤트 구간 한정 개입은 이번 평가에 포함하지 않는다.

모델 revision `962318cec55ac10993ff0f5f43eda9a270b4c873`, custom-code revision
`47a0ec7fc4ec123775a391911046cf33cf9ed83f`, SAE SHA-256
`18443083d320d2ad3c607431f307697639966ad92075bdfe031398557f3ba9d6`를 고정한다.
새 activation 수집이나 SAE 재학습은 하지 않는다.

**해석 범위:** 후보 선정·평가·기존 SAE 학습에 같은 500개가 겹친다. 미평가 데이터 일반화나
새 지표의 독창성을 검증하는 실험이 아니다. 과거 수집 당시 초기 상태/생성 설정의 누락된 증거도
이번 재실행으로 복구되지 않는다. Head는 8개 query, Event는 전체 시계열의 기존 token Top-64
자료를 쓰므로, episode 수가 같아져도 두 점수의 관측 범위까지 완전히 같지는 않다.

## 2. 비용과 통과 기준 — 먼저 확인

서로 다른 선정 feature가 U개면 본 평가는 **500 × (U+2)회**다.
U는 5~15개이므로 본 평가 범위는 3,500~8,500회이며, `plan`이 정확한 수를 출력한다.
과거 250개 점수에서 합집합은 8개였지만 **이번에도 8개라는 보장은 없다.**

본 평가와 별도로 **사전 동작 점검 10회**만 수행한다. 이전 402회 설계에서
본 평가와 중복되는 반복을 줄여, 별도 실행이 **392회 감소**했다.

- task 0의 trial 0/1을 원본 Baseline raw·어댑터 raw·raw 반복·identity에서 수행: 8회.
- 같은 2개 trial에서 α=0 hook이 실제 활성 feature를 변경했는지 확인: 2회.
- 이 두 상태에서 action trace·성공 여부를 **정확히** 비교한다. 불일치 시 중단한다.
- 이는 첫 task의 짧은 동작 점검일 뿐, task 전환이나 전체 Baseline 동등성을 검증한 것은 아니다.
- **본 평가의 raw 500회·identity 500회가 전체 상태에서 정확히 일치해야 feature 평가를 시작한다.**
  이 1,000회는 본 평가 예산에 이미 포함되어 있으므로 추가 비용이 아니다.

과거 RTX4090 로그 기준, 합집합이 8개라면 행동 평가만 대략 60시간 수준의 계획이다.
사전 검증·점수 계산·다운로드·재시도는 별도다. **현재 코드의 실제 GPU 소요 시간은 아직 측정하지 않았다.**
고성능 GPU여도 simulator/CPU 병목 때문에 가격에 비례해 빨라진다고 가정하면 안 된다.
한 번에 전체 비용을 승인하지 말고, 점수 계산 후 정확한 계획과 사전 검증 결과를 먼저 확인한다.

> 사전 점검은 **첫 task의 두 상태에서 Baseline 원본과 어댑터의 일치**를 확인한다.
> 이후 전체 500개에서 검증하는 것은 **본 실행의 raw와 identity 일치**다.
> 다른 GPU·라이브러리로 수행한 과거 팀원 성공률과 숫자까지 같아진다는 보장은 아니다.
> 팀원이 실제 사용한 Baseline commit/backend가 아래 고정값과 다르면 먼저 합의해야 한다.

## 3. 새 RunPod 준비

지원되는 RTX4090 런타임을 사용한다. 예전 Torch 2.2 이미지로 RTX5090을 사용하면 안 된다.
아래 작업은 **tmux 안에서** 수행한다. 노트북 연결이 끊겨도 Pod과 tmux 프로세스가 살아 있으면 계속된다.

```bash
event-sae-init
event-sae-verify --require-gpu
tmux new -s team500
```

아래 clone 명령은 검토 후 새 브랜치가 GitHub에 push된 다음 사용한다.

```bash
cd /workspace
git clone --branch research/team-baseline-500-top5 \
  https://github.com/jiyeon-yoon/Event-SAE-HeadRank.git Event-SAE-Team500
git clone https://github.com/jiyeon-yoon/Event-SAE-Baseline.git Event-SAE-TeamBaseline
git -C /workspace/Event-SAE-TeamBaseline switch --detach 56e9f012aa88532f9880259b371d9300a8013b9e
cd /workspace/Event-SAE-Team500
git rev-parse HEAD
git status --short
```

마지막 `git status --short`에서 아무것도 표시되지 않아야 한다.

```bash
CLI=scripts/openvla/headteam500.py
R=/workspace/head-team500-v1
B=/workspace/Event-SAE-TeamBaseline
export HF_HOME=/workspace/cache/head-team500-hf
export HF_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE
export PYTHONHASHSEED=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TF_FORCE_GPU_ALLOW_GROWTH=true
set -o pipefail
df -h /workspace
python -c 'import importlib.util; assert importlib.util.find_spec("flash_attn") is None, "STOP: agreed SDPA runtime must not include flash_attn"; print("SDPA_RUNTIME_PACKAGE_CHECK_OK")'
```

기존 데이터 약 281GiB에 모델·cache·결과 저장 공간이 추가로 필요하다. 여유 공간을 확인한다.
본 설정은 **SDPA**를 기대한다. stock Baseline은 `flash_attn`이 설치되어 있으면 다른 backend를
시도하므로 그 경우에는 사전 단계가 중단된다. 실행 도중 임의로 패키지나 backend를 바꾸지 않는다.

## 4. 기존 입력 내려받기·검증

공개 파일도 rate limit이 발생하면 먼저 로그인한다. 토큰은 터미널에만 입력한다.

```bash
HF_HUB_OFFLINE=0 hf auth login
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python "$CLI" download --execute
python "$CLI" merge
python "$CLI" verify-baseline --baseline-root "$B"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python "$CLI" setup --run-root "$R" --baseline-root "$B" --allow-libero
```

다운로드는 모델·SAE·5개 Spatial 데이터셋·공개 Event 결과의 고정 revision을 사용한다.
`processing_prismatic.py`의 중첩 cache 참조도 전용 HF cache에 고정한다.
setup은 실제 모델 shard 4개의 내용 SHA와 현재 LIBERO 초기 상태를 검사한다.
기존 경로를 덮어쓰지 않으며, 다른 설정으로 다시 실행하려면 새 `R`을 사용한다.

## 5. 새 500개 Head cache와 점수 계산

```bash
CFG="$R/head-config.yaml"
MODEL=/workspace/head-inputs/openvla-spatial
EVENT=/workspace/head-inputs/event-public/scores/event_feature_scores.pt

python scripts/openvla/headsetup.py --config "$CFG" --snapshot "$MODEL" --export
python scripts/openvla/headinputs.py --config "$CFG" --allow-simulator
python scripts/openvla/output_head_sensitivity.py validate \
  --config "$CFG" --allow-model-execution
```

여기서 `status: passed`를 확인한다. 이 단계의 소수 입력은 **출력층 수치 검증용**이며,
팀 Baseline의 행동 평가가 아니다. 기존 capture 코드의 fixture seed 방식도 본 평가에는 사용하지 않는다.

```bash
python "$CLI" prepare-cache --head-config "$CFG" --execute
python -u "$CLI" score --head-config "$CFG" --event-scores "$EVENT" \
  --output "$R/smoke.json" --smoke --execute 2>&1 | tee "$R/smoke.log"
python -u "$CLI" score --head-config "$CFG" --event-scores "$EVENT" \
  --output "$R/full-scores.json" --execute 2>&1 | tee "$R/score.log"
```

`prepare-cache`는 28,000 readout을 만든다. `smoke`는 개발 점검용이므로 본 평가를 승인할 수 없다.
`full-scores.json`은 500개 전체에서 다시 계산한다. 예전 250개 점수 파일을 가져오면 중단한다.
이번 전용 계산은 사용하지 않는 방향 대조 4개를 생략한다. CPU FP32/BF16 회귀 테스트에서
기존 Mean/Tail 값과 정확히 같음을 확인했지만, 실제 GPU 시간은 smoke 로그로 먼저 측정한다.

## 6. 선정·평가 계획 고정

```bash
python "$CLI" plan --scores "$R/full-scores.json" \
  --cases "$R/inputs/cases.json" --runtime-config "$R/inputs/runtime-config.json" \
  --output "$R/plan.json"
python "$CLI" verify-plan --plan "$R/plan.json"
PLAN_HASH=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["plan_hash"])' "$R/plan.json")
CODE_REV=$(git rev-parse HEAD)
printf 'PLAN_HASH=%s\nCODE_REV=%s\n' "$PLAN_HASH" "$CODE_REV"
```

**각 Top-5·합집합 수·`total_rollouts`를 확인한 다음 진행한다.** 이미 알고 있는 feature를 임의로 빼지 않는다.
계획 파일을 직접 수정하거나 결과를 보고 후보를 교체하면 안 된다.

## 7. GPU 사전 동작 점검 — 별도 10회

먼저 비용 없는 미리보기:

```bash
python "$CLI" preflight --plan "$R/plan.json" \
  --output-root "$R/preflight" --baseline-root "$B"
```

실제 실행:

```bash
python -u "$CLI" preflight --plan "$R/plan.json" \
  --output-root "$R/preflight" --baseline-root "$B" \
  --execute --max-rollouts 10 --approved-plan-hash "$PLAN_HASH" \
  --expected-code-revision "$CODE_REV" 2>&1 | tee "$R/preflight.log"
```

`status: passed`와 `$R/preflight/preflight.json`이 생성되어야 한다.
인증서는 10회 점검 전용 v2이며 전체 task 검증을 의미하지 않는다.
이전 v1/402회 인증서는 재사용하지 않는다. 이미 예전 코드로 점검한 경우 새 코드에서 새 출력 폴더를 쓴다.
불일치 시 허용 오차를 늘려 통과시키지 않는다. 실패 원인과 `child.stdout.log`부터 확인한다.
새 GPU·패키지 변경·코드 변경 후에는 이전 인증서를 재사용하지 않는다.

## 8. 본 평가 — raw/identity부터

첫 실행은 두 조건만 한다. 각 조건은 500개를 처음부터 수행한다.

```bash
python -u "$CLI" run --plan "$R/plan.json" --output-root "$R/runs" \
  --baseline-root "$B" --preflight "$R/preflight/preflight.json" \
  --execute --approved-plan-hash "$PLAN_HASH" --expected-code-revision "$CODE_REV" \
  --max-new-conditions 2 2>&1 | tee "$R/run-controls.log"
```

raw/identity가 전체 500개에서 정확히 일치하면, 나머지 조건을 실행한다.

```bash
python -u "$CLI" run --plan "$R/plan.json" --output-root "$R/runs" \
  --baseline-root "$B" --preflight "$R/preflight/preflight.json" \
  --execute --approved-plan-hash "$PLAN_HASH" --expected-code-revision "$CODE_REV" \
  2>&1 | tee "$R/run-features.log"
```

완료된 **조건 전체**는 재사용한다. 중간에 끊긴 조건의 완료 episode를 건너뛰지는 않는다.
기본 재시도 예산은 0이다. 실패한 조건은 자동 재시도하지 않으며, 처음 실행할 때 별도로 승인한
`--retry-budget-conditions N`만큼만 500회 단위 재실행을 허용한다. 이미 시작한 journal의 예산을
나중에 늘릴 수 없다. 재시도 시에는 같은 예산과 `--retry-incomplete`가 모두 필요하다.

## 9. 진행 확인·분석

재접속한 터미널에서는:

```bash
cd /workspace/Event-SAE-Team500
CLI=scripts/openvla/headteam500.py
R=/workspace/head-team500-v1
ps -eo pid,etime,args | grep '[h]eadteam500.py run'
python "$CLI" status --plan "$R/plan.json" --output-root "$R/runs"
tmux attach -t team500
```

실행 중 journal이 잠겨 있으면 `busy/locked`가 나올 수 있다. 이를 이유로 프로세스를 종료하거나
lock 파일을 지우지 않는다. 상위 로그는 조건 단위로 갱신되며, 실제 episode 로그는 아래에서 찾는다.

```bash
find "$R/runs" -name child.stdout.log -print
```

최신 실행 중인 경로를 골라 `tail -f /위에서/찾은/child.stdout.log`로 확인한다.
모든 조건이 완료된 뒤 CPU 분석:

```bash
python "$CLI" analyze --plan "$R/plan.json" --output-root "$R/runs" \
  --scores "$R/full-scores.json" --output "$R/analysis-v1"
sed -n '1,220p' "$R/analysis-v1/report.md"
```

보고서는 signed 성공률 하락·성공→실패/실패→성공·Top-5 평균·방법 간 차이·고정 평가 후보군의
Spearman을 기록한다. 과제 안에서 같은 case 인덱스를 공유해 2,000회 재표집한다.
Top-5 합집합은 대표성 있는 전체 feature 표본이 아니므로 전 feature 예측 정확도로 해석하지 않는다.

## 10. Pod 종료 전 백업

아래는 **완료 후 사용자가 실행하는 업로드 명령**이다. 모델·원본 281GiB는 고정 공개 경로에서
다시 내려받는다. 결과·cache·수치검증·계획·실행 증거·분석·양쪽 소스 코드를 보관한다.

```bash
cd /workspace/Event-SAE-Team500
BACKUP=$(mktemp -d /workspace/team500-backup.XXXXXX)
git archive --format=tar.gz HEAD -o "$BACKUP/source-code.tar.gz"
git -C /workspace/Event-SAE-TeamBaseline archive --format=tar.gz HEAD \
  -o "$BACKUP/baseline-source.tar.gz"
git rev-parse HEAD > "$BACKUP/source-commit.txt"
git -C /workspace/Event-SAE-TeamBaseline rev-parse HEAD > "$BACKUP/baseline-commit.txt"
python -m pip list --format=json > "$BACKUP/python-packages.json"
nvidia-smi > "$BACKUP/nvidia-smi.txt"
tar -czf "$BACKUP/experiment.tar.gz" -C /workspace head-team500-v1
(cd "$BACKUP" && sha256sum *.tar.gz *.txt *.json > SHA256SUMS)
(cd "$BACKUP" && sha256sum -c SHA256SUMS)
```

```bash
REPO=jiyeony/event-sae-head-full10-results
REMOTE="followups/team500-$(date -u +%Y%m%dT%H%M%SZ)"
HF_HUB_OFFLINE=0 hf upload "$REPO" "$BACKUP" "$REMOTE" --repo-type dataset
HF_REV=$(HF_HUB_OFFLINE=0 python -c 'import sys;from huggingface_hub import HfApi;print(HfApi().repo_info(sys.argv[1],repo_type="dataset").sha)' "$REPO")
CHECK=$(mktemp -d /workspace/team500-check.XXXXXX)
HF_HUB_OFFLINE=0 hf download "$REPO" --repo-type dataset --revision "$HF_REV" \
  --include "$REMOTE/*" --local-dir "$CHECK"
cmp "$BACKUP/SHA256SUMS" "$CHECK/$REMOTE/SHA256SUMS"
(cd "$CHECK/$REMOTE" && sha256sum -c SHA256SUMS)
printf 'REPO=%s\nHF_REV=%s\nREMOTE=%s\n' "$REPO" "$HF_REV" "$REMOTE"
```

원격 파일을 실제로 복원해서 완료 상태도 확인한다. 새 모델·GPU 추론은 하지 않는다.

```bash
mkdir -p "$CHECK/source" "$CHECK/restored"
tar -xzf "$CHECK/$REMOTE/source-code.tar.gz" -C "$CHECK/source"
tar -xzf "$CHECK/$REMOTE/experiment.tar.gz" -C "$CHECK/restored"
python "$CHECK/source/scripts/openvla/headteam500.py" status \
  --plan "$CHECK/restored/head-team500-v1/plan.json" \
  --output-root "$CHECK/restored/head-team500-v1/runs" \
  --scores "$CHECK/restored/head-team500-v1/full-scores.json"
```

본 평가의 모든 조건 완료, 다시 내려받은 모든 파일의 SHA 일치, 복원 후 `status: completed`를
확인한 뒤 Pod을 종료한다. 이 문서의 백업 명령은 기본 `R=/workspace/head-team500-v1` 기준이다.

## 구현과 아직 검증하지 않은 항목

- 기존 HeadRank 실행기나 Baseline 소스는 변경하지 않았다. 이번 전용 코드·CLI·출력 경로로 분리했다.
- 로컬 CPU·모의 실행 회귀 테스트를 수행했다. 실제 GPU·LIBERO의 10회 사전 점검과
  본 평가 raw/identity 각 500회 검증은 아직 하지 않았다.
- 기존 layer31 post-MLP residual hook을 모든 token·action step에 적용한다.
- 중단 시 case 단위로 재개하지 않는다. seed뿐 아니라 RNG·환경의 실행 이력도 맞추기 위해 조건 처음부터 재시작한다.
