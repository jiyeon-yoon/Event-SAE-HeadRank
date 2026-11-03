# Taylor 비교 — 새 RunPod 실행 안내

## 범위와 현재 상태

- 전용 브랜치: `research/taylor-comparison`.
- 전용 로컬 작업 폴더: `Event-SAE-Taylor`. 기존 HeadRank/HeadCommand/Baseline 작업 폴더를 바꾸지 않는다.
- 기존 OpenVLA, Layer31 SAE, discovery 250 episode의 lossless original-forward cache를 재사용한다.
- Taylor 점수는 전체 cache에서 활성인 **6,009 feature / 14,000 readout**에 계산한다.
- 기존 Mean/Tail Head-KL·Event-max·활성 빈도·편집량은 같은 표본의 고정 점수를 읽는다. 재계산한 것처럼 표시하지 않는다.
- 기존 두 행동 평가 집단(각16 feature×200 case)은 **분리**해서 비교한다.
- **새 로봇 rollout 0회, SAE 학습 0회.** 현재 GPU 검증은 저장된 입력으로 모델 추론만 한다.
- 실제 새 GPU 실행은 아직 하지 않았다. 아래 smoke를 통과한 후에만 full을 실행한다.
- 로컬 CPU 검증: Taylor 신규 테스트와 기존 회귀 검사를 포함해 **1,051개 통과**. GPU 동작 검증을 대신하지 않는다.
- 새 Pod에서는 GitHub의 `research/taylor-comparison` 브랜치를 받는다. **게시된 코드를 실행할 때는 1번부터 시작**한다.

필수 GPU 연산은 Taylor 계산과 현재 장치의 수치 검증이다. 281 GiB/369개 원본 activation은 받지 않는다.
모델 cache와 복사본, 복원본을 포함해 **60 GiB 이상 여유 공간**을 권장한다. 기존 runtime의 RTX4090/L40/A40 등에서 먼저 smoke한다.
RTX5090 같은 Blackwell GPU는 기존 Torch2.2 runtime과 호환되지 않으므로 별도 환경 검증 없이는 사용하지 않는다.

## 점수의 정확한 의미

원래 residual `h`에서 feature 기여 `z_j*d_j`만 제거한다고 할 때:

```text
y = 현재 검증된 native BF16 head의 full-vocabulary greedy ID
C(h) = -log p_FP32(y | h)
Taylor_j = 동일 task → episode → step → 7개 출력 위치 가중 평균
           abs(z_j * dot(d_j, gradient_h C(h)))
```

- `d_j`는 bias를 제외한 **실제 decoder vector**이며 단위 벡터로 정규화하지 않는다.
- 원래 residual을 SAE reconstruction으로 교체하거나 SAE를 다시 encode하지 않는다.
- 비활성 위치는 0으로 두고 전체 표본을 분모로 사용한다.
- target ID는 과거 실제 생성 token으로 단정하지 않는다. target을 edit 후 다시 선택하지도 않는다.
- 미분은 같은 frozen weights와 RMSNorm eps를 사용하는 **연속 FP32 surrogate**다. BF16 반올림 함수의 정확한 미분이 아니다.
- native/FP32 logits 차이와 argmax mismatch, decoder affine consistency를 기록한다.
- 이것은 성공률 loss의 기울기가 아니라 **원래 선택한 출력 token의 NLL 민감도**다.

비교군은 [Molchanov et al., ICLR2017 Taylor pruning](https://arxiv.org/html/1611.06440v2#S2.SS2)의
기준을 SAE feature에 적용한 것이다. 새로운 Taylor 원리 또는 CNN pruning 실험 전체의 재현이라고 주장하지 않는다.

## 0. 로컬에서 검토 후 commit·push

아래는 코드를 추가 수정한 개발자의 게시 절차다. 이미 게시된 코드를 새 Pod에서 실행하려면
이 단계는 생략하고 1번부터 시작한다. 로컬 `Event-SAE-Taylor` 폴더의 터미널에서 실행한다.

```bash
git branch --show-current
git diff --stat
git status --short

git add README.md docs/run_taylor_comparison.md
git add event_sae/research/head_taylor scripts/openvla/headtaylor.py
git add tests/test_head_taylor_*.py
git commit -m "feat: add isolated Taylor importance comparison"
git push -u origin research/taylor-comparison
git rev-parse HEAD
```

마지막 40자 commit을 기록한다. 새 Pod의 clone HEAD가 이 값과 같은지 확인한다.

## 1. 새 Pod: runtime과 코드 준비

이전과 같은 runtime image를 사용한다.

```text
ghcr.io/jiyeon-yoon/event-sae-runtime@sha256:ef895731ffd73985e1183c964b479bdcc7e97dfeabea6c8777b903c54ade1405
```

```bash
event-sae-init
event-sae-verify --require-gpu
tmux new -s taylor
```

이후 작업은 tmux 안에서 한다. Pod은 켜둬야 하지만 노트북 연결은 종료할 수 있다.

```bash
cd /workspace
git clone --branch research/taylor-comparison \
  https://github.com/jiyeon-yoon/Event-SAE-HeadRank.git Event-SAE-Taylor
cd /workspace/Event-SAE-Taylor
git rev-parse HEAD
git status --short
```

로컬에서 게시한 commit과 같은지 확인한다. 다르면 실험을 시작하지 않는다.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/workspace/cache/taylor-hf
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

STEP=/workspace/head-taylor-v1
CLI=scripts/openvla/headtaylor.py
mkdir -p "$STEP/outputs"
git rev-parse HEAD > "$STEP/source-commit.txt"

python -m pytest -q tests/test_head_taylor_*.py
```

테스트가 실패하면 다음 단계로 진행하지 않는다.

## 2. 최소 입력 다운로드·복원

공개 파일이지만 429가 발생하면 `hf auth login`으로 로그인한다. 토큰은 채팅·소스·로그에 넣지 않는다.

```bash
python "$CLI" download --workspace "$STEP"
python "$CLI" download --workspace "$STEP" --execute
```

첫 명령은 다운로드 계획만 출력한다. 두 번째가 pinned revision 다운로드·hash 검증을 수행한다.
processor custom code 3개와 nested cache reference도 함께 준비하므로 과거의 `processing_prismatic.py` 누락 오류를 예방한다.
동일 파일은 재사용하고, 다른 파일이 이미 있으면 덮어쓰지 않는다.

```bash
python scripts/openvla/headfollowup.py stage1-restore \
  --archive "$STEP/bundle/experiment.tar.gz" \
  --manifest "$STEP/bundle/manifest.json" \
  --output "$STEP/restored"

python scripts/openvla/headfollowup.py stage1-setup \
  --restored "$STEP/restored" \
  --sae "$STEP/sae/trainer_0/ae.pt" \
  --snapshot "$STEP/model" \
  --output "$STEP/config"
```

restore/setup은 새 경로에 한 번만 수행한다. 재접속했다면 기존 파일을 삭제하거나 다시 만들지 말고 아래 변수만 복원한다.

```bash
CFG="$STEP/config/action-runtime.yaml"
REF="$STEP/restored/results/full10-followup-v1"
BASE="$STEP/baselines/full-scores.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 3. 현재 GPU 수치 검증

```bash
python scripts/openvla/output_head_sensitivity.py validate \
  --config "$CFG" --allow-model-execution
```

`"status": "passed"`가 필요하다. 과거 archive의 validation을 새 GPU 결과로 재사용하지 않는다.
실패하면 허용오차·hash·검증 guard를 풀지 말고 오류를 확인한다.

## 4. Smoke + 같은 GPU의 계산 시간 확인

```bash
set -o pipefail
python -u "$CLI" score \
  --config "$CFG" \
  --reference-result "$REF" \
  --baseline-scores "$BASE" \
  --output "$STEP/outputs/smoke.json" \
  --benchmark-kl --execute \
  2>&1 | tee "$STEP/outputs/smoke.log"
```

- task당 1 discovery episode의 전체7위치·전체8시점: **560 readout**.
- 해당 smoke의 모든 활성 feature에 대해 계산한다.
- 같은 readout·GPU에서 exact Head-KL도 계산하여 timing을 기록한다.
- `development_only: true`다. 기존 full 점수와 표본이 다르므로 이 파일로 행동 비교 보고서를 만들지 못하게 차단한다.
- `diagnostics`에 실제 시간·native/FP32 차이·decoder 검사가 나온다. `benchmark_exact_kl` 상세 시간은 smoke JSON에 있다.
- 한 번의 smoke timing은 예비 측정이다. 이를 전 GPU·전 표본의 확정 속도 향상으로 해석하지 않는다.

Smoke가 통과하고 예산 내에서 완료 가능하다고 판단한 후 full로 진행한다.

## 5. 전체 Taylor 점수 계산

```bash
python -u "$CLI" score \
  --config "$CFG" \
  --reference-result "$REF" \
  --baseline-scores "$BASE" \
  --output "$STEP/outputs/full-scores.json" \
  --full-readouts --execute \
  2>&1 | tee "$STEP/outputs/full-score.log"
```

100 readout마다 진행률·경과 시간·예상 잔여 시간을 출력한다. 최종 기대값:

```text
status: scored
sampling_mode: full
development_only: false
features: 6009
diagnostics.readouts: 14000
diagnostics.active_pairs: 777073
new_rollouts: 0
```

현재 GPU full 소요 시간은 미실측이다. 옛 HeadRank의 2시간38분에는 방향 대조4개가 포함되므로 이번 시간으로 그대로 환산하지 않는다.
중단되면 완성된 출력으로 취급하지 않는다. 이 버전은 readout 중간 재개를 지원하지 않으며 다시 실행해야 한다.
완료 파일이 이미 있으면 새 이름을 사용한다. 기존 파일을 덮어쓰지 않는다.

## 6. 기존 두 행동 집단과 분리 비교

```bash
python "$CLI" compare \
  --scores "$STEP/outputs/full-scores.json" \
  --panel "original_full10=$REF/analysis/paired_effects.json" \
  --panel "new_headrank=$STEP/new-panel/analysis.json" \
  --output "$STEP/outputs/comparison-v1"

sed -n '1,200p' "$STEP/outputs/comparison-v1/report.md"
```

읽을 것:

1. `Net-loss Spearman`: 순손상 `(성공→실패 − 실패→성공)/case`와 점수 순위의 관계.
2. `Outcome-flip Spearman`: 성공 여부 변화율 `(성공→실패 + 실패→성공)/case`와 관계.
3. `Original Top-K / Coverage`: 전체 점수의 원래 Top3가 실제 평가되었는지.
4. `undefined`: 미평가 Top3가 있으면 결과를 만들거나 측정된 하위 feature로 대체하지 않는다.

두 집단은 raw160/200, raw161/200을 각각 유지한다. 하나의 표본으로 합치지 않는다.
통계는 **기존 선정 panel에서의 사후 기술통계**다. 현재 버전은 CI·p-value·새 feature 일반화 검정을 수행하지 않는다.
새 Taylor 계산은 historical seed/수집 이력 누락이나 기존 SAE의 학습 데이터 겹침을 복구하지 않는다.
또한 평균·Tail KL은 과거 BF16 점수이며, 현재 FP32 Taylor와 수치 정의가 다르다는 것을 보고서에 명시한다.

## 7. 결과 백업 — Pod 종료 전

아래 archive는 재분석에 필요한 결과·두 행동 집단·baseline·소스·검증을 담는다.
모델15GB·원본 cache 전체는 넣지 않고 위 pinned 입력에서 다시 복원한다.
로그·환경 정보에 토큰/비밀이 없는지 먼저 확인한다.

```bash
python -m pip list --format=json > "$STEP/outputs/python-packages.json"
nvidia-smi > "$STEP/outputs/nvidia-smi.txt"
git status --short > "$STEP/outputs/git-status.txt"
test -z "$(git status --porcelain)" || { echo SOURCE_DIRTY_STOP; exit 1; }
git archive --format=tar.gz --output="$STEP/outputs/source-code.tar.gz" HEAD

tar -czf "$STEP/taylor-results.tar.gz" -C "$STEP" \
  outputs config source-commit.txt \
  baselines/full-scores.json new-panel/analysis.json \
  restored/results/full10-followup-v1/analysis/paired_effects.json

cd "$STEP"
sha256sum taylor-results.tar.gz > taylor-results.tar.gz.sha256
```

기존 결과 dataset에 새 하위 경로로 업로드한다. 기존 release를 바꾸지 않는다.
다운로드 때 켜둔 offline 설정을 해제하고 로그인한다.

```bash
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
hf auth login

REPO=jiyeony/event-sae-head-full10-results
REMOTE="followups/taylor-$(date -u +%Y%m%dT%H%M%SZ)"
hf upload "$REPO" "$STEP/taylor-results.tar.gz" \
  "$REMOTE/taylor-results.tar.gz" --repo-type dataset
hf upload "$REPO" "$STEP/taylor-results.tar.gz.sha256" \
  "$REMOTE/taylor-results.tar.gz.sha256" --repo-type dataset
```

원격에서 다시 받아 검증한다. 아래 revision은 **업로드 직후** 한 번 고정해 두 파일을 같은 revision에서 받는다.

```bash
HF_REV=$(python -c 'from huggingface_hub import HfApi;print(HfApi().repo_info("jiyeony/event-sae-head-full10-results",repo_type="dataset").sha)')
CHECK=$(mktemp -d /workspace/taylor-verify.XXXXXX)
hf download "$REPO" "$REMOTE/taylor-results.tar.gz" \
  "$REMOTE/taylor-results.tar.gz.sha256" \
  --repo-type dataset --revision "$HF_REV" --local-dir "$CHECK"

(cd "$CHECK/$REMOTE" && sha256sum -c taylor-results.tar.gz.sha256)
printf 'REPO=%s\nHF_REV=%s\nREMOTE=%s\n' "$REPO" "$HF_REV" "$REMOTE"
```

`taylor-results.tar.gz: OK`와 원격 revision·경로를 기록한 뒤에만 Pod 종료를 판단한다.

## 재접속용 변수

```bash
cd /workspace/Event-SAE-Taylor
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/workspace/cache/taylor-hf
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
STEP=/workspace/head-taylor-v1
CLI=scripts/openvla/headtaylor.py
CFG="$STEP/config/action-runtime.yaml"
REF="$STEP/restored/results/full10-followup-v1"
BASE="$STEP/baselines/full-scores.json"
```

프로세스 확인: `ps -eo pid,etime,args | grep '[h]eadtaylor.py score'`.
tmux 세션 확인: `tmux ls`, 복귀: `tmux attach -t taylor`.
