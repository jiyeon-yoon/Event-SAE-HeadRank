# 출력층 민감도: 실행·복구 안내

현재 HeadRank 후속 개발은 **G. Tail Full10 + 보정 예측기**를 참고한다.
A~F는 기존 실험의 복구/실행 기록이며 기존 3,600회를 다시 실행하라는 지시가 아니다.

연구 내용·결과·데이터별 역할은 [experiment.md](experiment.md)를 참고한다.
**이미 완료된 결과만 확인하려면 A만 실행한다. GPU Pod을 새로 만들 필요가 없다.**
Dr.VLA 후속 비교는 아래 D의 별도 규칙이다. 기존 B의 3,600회 실행을 다시 시작할 필요는 없다.

## A. 공개 결과 다운로드·검증·복구 — GPU 불필요

Python 3.10 이상과 `hf` CLI를 사용한다. 공개 결과는 로그인 없이 다운로드할 수 있다.
CLI가 없으면 별도 환경에 `python -m pip install huggingface_hub`로 설치한다.

```bash
REPO=jiyeony/event-sae-head-full10-results
HF_REV=69c33d6a943c253dc97cf9a3968095276f7e9540
BUNDLE=./head-full10-release
export HF_HUB_OFFLINE=0

hf download "$REPO" --repo-type dataset --revision "$HF_REV" --local-dir "$BUNDLE"
python "$BUNDLE/tools/headbundle.py" verify --bundle "$BUNDLE"
```

정상: `status: verified`, `total_rollouts: 3600`, `conditions: 18`.
archive 약 1.53 GB와 임시 압축 해제 공간이 필요하다. 검증은 tensor/pickle을 실행하거나
모델을 재추론하지 않는다. 저장된 파일 해시·case·행동 기록·성공률 하락을 대조한다.
bootstrap 신뢰구간 전체 재계산과 simulator 재실행을 뜻하지 않는다.

보고서 바로 읽기:

```bash
sed -n '1,180p' "$BUNDLE/reports/report.md"
sed -n '1,180p' "$BUNDLE/reports/task_groups_v1.md"
```

`reports/episodes.csv`는 3,600회 평가 목록, `reports/summary.json`은 방법별 수치·비교 CI·시간이다.
전체 원자료까지 열려면 **아직 없는 새 목적지**에 복구한다.

```bash
RESTORED=./head-full10-restored
python "$BUNDLE/tools/headbundle.py" restore --bundle "$BUNDLE" --destination "$RESTORED"
python "$BUNDLE/tools/headbundle.py" inspect --root "$RESTORED/results/full10-followup-v1"
```

복구 구조:

```text
head-full10-restored/
  results/full10-followup-v1/  # 18개 run, 점수, head/readout, parity, plan, 분석, 로그
  inputs/full10/              # split, source/generation/eval, runtime, discovery Event score
  config/                    # 원본 YAML와 plan hash에 연결된 normalized JSON
  source/experiment-code.tar.gz
  sources/                   # 원본 다운로드 receipt와 pinned 입력 참조
  environment/               # 백업 시점 package/GPU 기록
```

원래 JSON/YAML의 `/workspace/...` 경로는 증거 보존을 위해 바꾸지 않는다. 다른 경로에서
원래 `analyze`까지 실행하려면 당시 코드·의존성·입력을 원래 경로로 복원해야 한다.
단순 열람/검증은 위 portable 도구만으로 된다. **pilot용 `headrestore.py`와 혼동하지 않는다.**

## B. 점수 계산부터 실제 rollout까지 새로 실행 — GPU·대용량 입력 필요

이번 40.88시간짜리 실험을 다시 계산하려는 경우에만 진행한다. 원본 5개 약 281 GiB,
OpenVLA 약 15 GB, SAE, Event 입력 및 결과 공간이 추가로 필요하다. 아래는 **기존 산출물이
없는 새 `/workspace`**를 전제로 한다. 복구한 결과 위에 새로운 실험을 덮어쓰지 않는다.
다른 모델/설정의 새 연구는 별도 protocol/output으로 설계해야 한다.

### B1. 실행 환경과 코드

기존 기록: RTX 4090 1장, 400 GB 디스크 설정, 아래 runtime 이미지.
현재 가용성·가격이나 새 장비에서의 수치 일치를 보장하는 목록은 아니다.

```text
ghcr.io/jiyeon-yoon/event-sae-runtime@sha256:ef895731ffd73985e1183c964b479bdcc7e97dfeabea6c8777b903c54ade1405
```

```bash
event-sae-init
event-sae-verify --require-gpu
cd /workspace
git clone https://github.com/jiyeon-yoon/Event-SAE-Head.git
cd /workspace/Event-SAE-Head
git checkout --detach 8924a8d9b62edb16d261785cbc2124729017f2b1
git status --short
```

마지막 출력이 비어 있어야 한다. 위 commit은 실제 full10 실행 코드다.
최신 main의 문서/백업 도구 revision과 구분한다. 이미 clone된 저장소라면 위 clone을
반복하거나 로컬 수정을 덮어쓰지 말고 상태부터 확인한다.

```bash
tmux new -s head-full10
```

tmux 안에서 다음 변수를 설정한다. 재접속/새 shell에서는 다시 설정한다.

```bash
cd /workspace/Event-SAE-Head
export HF_HOME=/workspace/cache/head-full10-hf
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CFG=configs/local/head-full10.yaml
OUT=/workspace/event-sae-head-results/full10-followup-v1
META=/workspace/head-inputs/full10
MODEL=/workspace/head-inputs/openvla-spatial
EVENTS=/workspace/head-inputs/event-public
CLI=scripts/openvla/output_head_sensitivity.py
```

### B2. 입력·분리·검증

다운로드 전 필요하면 `HF_HUB_OFFLINE=0 hf auth login`으로 로그인한다. 토큰은 로그인
입력창에만 넣는다. 전용 HF cache를 사용하며, 기존 다른 모델의 `main` 참조를 덮어쓰지 않는다.
**한 명령이 성공한 뒤 다음 명령으로 넘어간다.**

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python scripts/openvla/headfull.py download --execute
python scripts/openvla/headfull.py merge
python scripts/openvla/headfull.py setup --allow-libero
```

정상: `downloaded` → `merged_and_verified` → `configured`.
369 dense shards, 500 episodes, discovery 250 / validation 50 / evaluation 200, `mode: followup`.
입력 revision은 실행 commit의 downloader에 고정되어 있다. 원본 상태의 과거 hash가
없다는 한계는 유지되며, 현재 상태 hash를 과거 수집의 증거로 바꾸지 않는다.

```bash
python scripts/openvla/headsetup.py --config "$CFG" --snapshot "$MODEL" \
  --metadata-output "$META/head.json" --export
python "$CLI" prepare --config "$CFG"
python scripts/openvla/headinputs.py --config "$CFG" \
  --output "$META/runtime.pt" --allow-simulator
python "$CLI" validate --config "$CFG" --allow-model-execution
```

`validate`의 `status: passed`가 필요하다. 실패하면 tolerance/manifest를 임의로 완화하지 않는다.
이미 존재하는 `runtime.pt`를 무작정 덮어쓰지 않는다. TensorRT/Gym 경고 자체와 최종 실패를 구분한다.

### B3. 점수와 실행 계획

```bash
python "$CLI" score --config "$CFG"
python scripts/openvla/headevents.py --config "$CFG" --event-root "$EVENTS"
python "$CLI" plan --config "$CFG"
```

두 점수는 같은 discovery episode만 사용한다. `headevents`는 기존 cluster를 유지하고
discovery-only Event score를 재집계한다. 원래 full500 점수에 새 hash만 붙이는 방식이 아니다.
정상 계획은 `planned`, `selection_eval_overlap: false`, 최대 4,000회다.
공개 실험에서는 후보 중복 제거 후 16개 feature / 18조건 / 3,600회가 선택됐다.

```bash
PLAN="$OUT/plan/rollout_plan.json"
PLAN_HASH=$(python -c 'import json,sys;from event_sae.research.output_head.provenance import fingerprint;p=json.load(open(sys.argv[1]));h=p.pop("plan_hash");assert h==fingerprint(p);print(h)' "$PLAN")
CODE_REV=$(git rev-parse HEAD)
git status --short
LOG="$OUT/full10-run.log"
```

### B4. 계획·비용을 확인한 후에만 실행

아래부터 실제 장시간 GPU 평가가 시작된다. 기존 조건별 시간 합계는 약 40.88시간이지만
새 장비의 소요 시간/비용 보장은 아니다. 현재 plan과 Pod 요금을 확인한 뒤 실행한다.

```bash
nohup env PYTHONUNBUFFERED=1 python "$CLI" run --config "$CFG" --execute \
  --approved-plan-hash "$PLAN_HASH" --expected-code-revision "$CODE_REV" >> "$LOG" 2>&1 &
echo $!
tail -f "$LOG"
```

- `tail -f`의 Ctrl+C는 로그 보기만 종료한다. Pod 자체 종료는 실행을 중단한다.
- tmux 재접속: `tmux attach -t head-full10`. 실행 중 code/config/split/score를 바꾸거나 git pull하지 않는다.
- 완료 기준: 최종 `status: completed`와 계획의 `total_rollouts`가 일치.
- 같은 환경에서 완료된 조건은 재사용할 수 있지만, 미완료 조건은 episode 단위 자동 복구가 아니다.
  일부 파일이 남아 재개가 차단되면 로그를 보존하고 원인을 확인한다. 조건 폴더를 임의 삭제하지 않는다.

### B5. 분석

```bash
python "$CLI" analyze --config "$CFG"
python scripts/openvla/headgroups.py --config "$CFG"
sed -n '1,180p' "$OUT/analysis/report.md"
sed -n '1,180p' "$OUT/analysis/task_groups_v1.md"
```

`analysis.json`의 `topk_contrasts`는 방법 간 차이/CI, `paired_effects.json`은 같은
평가 case의 raw/edit 성공 여부다. subgroup에서도 기존 순위와 후보를 바꾸지 않는다.
과거 숫자와 일치하도록 후보·초기상태·seed를 사후 조정하지 않는다.

## C. 새 실행 결과 백업·공개 — 완료 후에만

기존 공개본은 이미 검증됐으므로 다시 업로드할 필요 없다. 아래는 새 실행의 백업 절차다.
**기존 공개본을 덮어쓰지 않도록 별도 결과 저장소를 사용한다.**

실험과 분석을 완료한 뒤에만 최신 백업 도구를 받는다. Git 작업물이 깨끗한지 먼저 확인한다.
백업 도구는 결과에 기록된 실행 commit의 소스를 따로 archive하므로 과거 실행 이력을 바꾸지 않는다.

```bash
cd /workspace/Event-SAE-Head
git status --short
git switch main
git pull --ff-only origin main
BUNDLE=/workspace/full10-public-release-v1
python scripts/openvla/headbundle.py pack --workspace /workspace --output "$BUNDLE"
python scripts/openvla/headbundle.py verify --bundle "$BUNDLE"
```

pack은 기존 실험 환경의 PyYAML이 필요하다. 출력 폴더는 새 경로여야 한다.
전체 결과/생성 입력/설정/실행 소스를 보존하지만 원본 dense·전체 모델·SAE는 pinned 참조로 남긴다.

```bash
export HF_HUB_OFFLINE=0
export REPO=jiyeony/event-sae-head-full10-results-rerun
hf auth login
python -c 'import os;from huggingface_hub import HfApi;HfApi().create_repo(os.environ["REPO"],repo_type="dataset",private=False,exist_ok=True)'
python -c 'import os;from huggingface_hub import HfApi;assert not HfApi().repo_info(os.environ["REPO"],repo_type="dataset").private'
hf upload "$REPO" "$BUNDLE" . --repo-type dataset
HF_REV=$(python -c 'import os;from huggingface_hub import HfApi;print(HfApi().repo_info(os.environ["REPO"],repo_type="dataset").sha)')
printf 'Published revision: %s\n' "$HF_REV"
```

`REPO`는 새 실험에 맞는 아직 사용하지 않은 이름으로 정한다. `pack`이 만드는 README의
다운로드 예시는 원래 공개 저장소를 가리키므로, 새 저장소의 안내에는 위 REPO/HF_REV를 명시한다.
원본 bundle 안 파일을 직접 고치면 checksum 검증이 깨진다.
private 저장소의 공개 여부를 자동 변경하지 않는다. 공유 전에 민감한 사용자 파일이 없는지도 확인한다.

실제 원격 archive를 다시 받아 검사한다. checksum 파일만 비교하는 것으로 끝내지 않는다.

```bash
df -h /workspace
CHECK=$(mktemp -d /workspace/full10-download-check.XXXXXX)
hf download "$REPO" --repo-type dataset --revision "$HF_REV" --local-dir "$CHECK" --force-download
python "$CHECK/tools/headbundle.py" verify --bundle "$CHECK"
```

`verified`와 plan의 조건/rollout 수가 일치한 뒤 HF revision을 보관한다.
**그 밖에 필요한 Pod 파일까지 백업했다면 terminate해도 된다.**

## D. Dr.VLA 비교 점검과 후속 규칙

### D1. 이번에 완료한 것과 결정

- 기존 결과 재분석 완료: [요약과 수치](experiment.md#22-2026-09-27-사후-재분석-큰-신호를-고르는-것뿐인가).
- Dr.VLA 원문과 저장 형식 대조, pinned 공개 파일 목록, 실제 TopK shard 1개 및
  Head readout cache 14개(14,000 readout)를 CPU로 검사했다. 원본 해시를 검증했고
  모델을 실행하거나 unrestricted pickle을 사용하지 않았다.
- [입력 점검 기록](../research_metadata/drvla_feasibility_20260927.json)과
  [비교 규칙 v1](../configs/research/openvla/drvla_comparison_protocol_v1.json)을 기록했다.
- **전면 재수집·OpenVLA/SAE 재학습은 보류한다.** 기존 dense 데이터로 필요한 시계열을
  만들 수 있는 경로가 있으므로, 먼저 그 경로의 계산량을 산정한다.

현재 Head cache는 250개 episode에서 **각 8 step만** 남겼다. 연속 활성화 횟수와 지속 길이를
그대로 계산할 수 없다. 별도 legacy TopK에는 전체 시계열이 있지만 token당 64개만 저장하며,
BatchTopK의 원래 forward grouping과 lossless 지원을 보증하지 않는다.
실제 Head cache에서 1,581/14,000 readout은 nonzero가 64개를 넘었고 최대 333개였다.
TopK 첫 shard 표본에는 50,187개 token row 중 532개가 저장한 64개 모두 양수였다.
이는 **누락된 값이 0이라는 보증이 없다는 점검 결과**이며 모든 legacy row가 틀렸다는 뜻이 아니다.
첫 legacy shard와 Head 표본은 이번 점검에서 동일 token의 직접 대조 표본을 확보하지 못했다.

기존 원본 dense 369개(약 280.85 GiB)와 index의 pinned remote inventory는 존재한다.
그 전체 tensor의 재다운로드·재인코딩은 이번 점검에 포함되지 않았다.

### D2. 고정한 비교 규칙

이 파일은 연구 규칙이며 `output_head_sensitivity.py --config`로 실행하는 config가 아니다.
과거 `rollout_plan.json`을 고치거나 기존 승인 hash로 새 실험을 실행하지 않는다.

1. **연구 대상 고정:** 같은 OpenVLA checkpoint, Layer 31 SAE, 기존 discovery 250 episode와
   feature ID를 사용한다. 기존 Head/Event 순위와 과거 결과는 수정하지 않는다.
2. **시계열 재구성:** 원본 dense를 original-forward 단위로 기존 SAE에 통과시킨 뒤,
   매 step의 7개 출력 readout latent를 평균한다. 모든 nonzero를 보존하고 step 누락은
   오류로 처리한다. hidden을 먼저 mean-pool해 SAE에 넣는 것과 혼동하지 않는다.
3. **Dr.VLA 기반 적용 버전:** 네 일반성 통계를 사용하되 `drvla_statistics_readout_recalibrated_v1`로
   이름을 구분한다. 원 논문의 mean-pooled-hidden SAE나 분류기 전체를 재현했다는 주장은 하지 않는다.
4. **스케일과 보정:** 현재 nonzero latent의 중앙값은 약 49.87이다(Head 표본 기준).
   원 논문과 입력 정규화가 다르므로 공개 계수를 무검증 상태로 주 비교에 쓰지 않는다.
   discovery 영상·활성 trace에서 일반적/episode-specific feature를 판정한 별도 라벨로
   분류기를 보정한다. 이전 16개 평가 feature는 보정 학습에서 제외하고, 성공률·Head 순위를
   라벨링이나 조정에 쓰지 않는다. 상세 라벨 목표와 고정 학습 설정은 JSON 규칙에 있다.
   라벨을 확보하지 못하면 합성 점수를 만들어 대신하지 않고 이 단계에서 멈춘다.
5. **선정:** Top-3, 동점 feature ID 오름차순. 기존 16개에서의 관계는 탐색적으로 먼저 보고한다.
   새로운 전체 Top-3가 미평가 feature를 포함하면 누락으로 표시하고 다른 후보로 대체하지 않는다.
6. **평가:** 동일 task/초기상태/seed/runtime의 paired 성공률 하락을 사용한다. 기존 결과 재사용은
   동일 protocol 확인 시에만 허용한다. 실제 새 Top-3가 모두 측정된 경우에만 전체 Top-3 성능을 보고한다.
7. **해석:** 기존 Head는 8 step, Dr.VLA 시계열은 전체 step이므로 입력 정보량·시간 비용 차이를
   공개한다. 일반성 라벨링 비용도 포함한다. “성공률을 더 떨어뜨렸다”를 “더 해석 가능하다”로 바꾸지 않는다.

위 보정 설정은 **이 프로젝트에서 고정한 적용 규칙**이며 Dr.VLA 원문의 학습 설정을
그대로 옮겼다는 뜻이 아니다. 기존 성공률을 이미 본 뒤의 follow-up이므로 사전등록 확증 실험으로 부르지 않는다.

### D3. 남은 일과 실행 경계

| 단계 | 현재 상태 | 새 로봇 rollout 필요? |
|---|---|---|
| 변경량 재분석·입력 감사·비교 규칙 | 완료 | 아니오 |
| 원본 dense → 전체 시계열 lossless latent | 미실행, 대용량 I/O 및 SAE 계산량 산정 필요 | 아니오 |
| Dr.VLA 통계·일반성 라벨·보정된 순위 | 미실행 | 아니오 |
| 기존 16개 결과와 연결 | Dr.VLA 순위 생성 후 가능 | 아니오 |
| 새 상위 후보 검증 | 후보 누락 수에 따라 별도 비용·승인 계획 | 경우에 따라 필요 |

**GPU Pod 생성, 281 GiB 자동 다운로드, 재수집, 재학습, 새 rollout은 이 문서만으로 실행하지 않는다.**
필요한 offline 재인코딩도 단순 JSON 재분석과 달리 계산 비용이 발생할 수 있다.

원본 tensor/forward mapping의 무결성 점검이 실패하거나 별도 prospective 검증의 필요성이
명확해질 때 새 데이터 수집을 다시 판단한다. 새 검증에서는 복원 가능한 초기상태 데이터와 hash,
seed·runtime·코드·모델 identity를 기록한다. SAE 학습/순위 계산/최종 평가를 먼저 분리한다.
해시만으로 누락된 과거 수집 상태를 복원할 수 없으며, 기존 SAE를 고정한 독립 검증도 가능하다.
OpenVLA 자체를 새로 학습할 필요는 없고 기존 팀 공용 데이터·SAE도 그대로 보존한다.

출처: [Dr.VLA v2](https://arxiv.org/html/2603.19183v2) §3 및 부록 A–C.

<a id="followup-all5"></a>

## E. 추가 ablation ①~⑤ — 기존 SAE 고정, 별도 브랜치

### E1. 보존 범위와 현재 개발 상태

- 기존 실험 코드 기준: `81487a07509ee10433c41c19d4bbcd079b9a4968`.
- 로컬 고정 태그: `freeze/head-before-drvla-20260927`.
- 후속 브랜치: `research/drvla-followup`. 2026-09-28 사용자 승인으로 이 브랜치만 commit·push한다.
  `main` 병합·갱신은 이번 승인 범위에 포함하지 않는다.
- 기존 `event_sae/research/output_head`, OpenVLA runner/hook 및 원 결과는 수정하지 않는다.
- 새 모듈은 `event_sae/research/head_followup`, 새 CLI는 `headfollowup.py`에만 추가한다.
- 미커밋 파일은 Git 브랜치에 영구 저장된 것이 아니다. 미커밋 상태로 `main`을 오가며 작업하지 않는다.

| 항목 | 로컬 구현 | 실제 입력/GPU에서 남은 일 |
|---|---|---|
| ① 변경량 대조 | 같은 hidden의 token row마다 anchor 제거 L2 크기로 다른 decoder/고정 Gaussian 방향 편집 및 새 hook | 새 조건의 실행·분석 연결, BF16 실현 변경량 오차 검증 |
| ② 억제 강도 | 원 산술을 유지하는 `alpha=0, .5, .75, 1` kernel/hook | GPU 일치 검증 및 전체 조건 실행 |
| ③ 행동 단계 | pre-action 대상 물체 상태에 따른 phase rule/gate/hook, 정보 없으면 오류 | LIBERO 물체별 관측 adapter, 영상 대조로 판별 기준 검증, 단계별 rollout |
| ④ action-token | pinned tokenizer/policy 매핑 확인, 조건부 KL+확률 질량+argmax 점수 CLI, 비교 보고서 | 현재 GPU 수치 검증 및 실제 점수 계산 |
| ⑤ Dr.VLA 기반 | 원본 dense→전체 readout latent CLI, 통계, 순위·성공률을 가린 검토 자료, 독립 라벨 보정·비교 CLI | 원본 복원/재인코딩, 영상 대조를 통한 실제 30개 라벨, 보정·순위 계산 |

**단위 테스트 통과는 실제 SAE/모델 numerical parity 또는 로봇 성능 검증이 아니다.**
2026-09-27 초기 core 개발 테스트 401개 통과 후, 1단계 CLI·안전장치 테스트 13개를 추가해
로컬 전체 테스트 **414개 통과**(follow-up CPU 테스트 총 58개 포함).
[개발·검증 기록](../research_metadata/head_followup_development_20260927.json)에 실제 완료/미완료를 구분했다.
후속 [1단계 작업 기록](../research_metadata/head_stage1_development_20260927.json)에는 실제 복원·점검과
아직 수행하지 않은 점수 계산을 따로 기록했다.
현재 `headfollowup.py`에는 `run` 명령을 제공하지 않는다. 원 CLI의 고정 조건을 완화하거나
옛 승인 hash로 새 조건을 실행하지 않는다. 새 결과는 별도 output에 저장한다.

### E2. 전체 10-task 설계와 사전 비용

설계 파일: [head_followup_all5.json](../configs/research/openvla/head_followup_all5.json).
설계 확인은 GPU·모델·다운로드 없이 가능하다:

```bash
python scripts/openvla/headfollowup.py design
```

Anchor는 기존 Head-KL Top-3 `24830, 18471, 7024` 그대로다. 성공률 하락이 없었던
`18471`도 유지한다. 기존 결과가 좋은 feature만 새로 골라 쓰지 않는다.
다만 기존 결과를 본 후의 추가 검증이므로 사전등록 확증 연구로 부르지 않는다.

| 조건 묶음 | 조건 수 | 조건당 평가 |
|---|---:|---:|
| raw + identity | 2 | 10 task × 20 초기상태 = 200 |
| 기존 Head/Event Top-3 합집합의 완전 제거 | 5 | 200 |
| anchor별 다른 SAE 방향 1 + Gaussian 방향 3 seed | 12 | 200 |
| anchor별 25% 억제/50% 억제 (`alpha=.75/.5`) | 6 | 200 |
| anchor별 접근/잡기/이동 단계에서만 완전 제거 | 9 | 200 |
| action-KL/Dr.VLA의 실제 Top-3 중 위와 중복되지 않는 feature | 0~6 | 200 |
| **합계** | **34~40** | **6,800~8,000 rollout** |

이는 **실행 승인 예산이 아닌 설계 추정치**다. 옛 결과의 자동 재사용 없이 산정했다.
기존 3,600회에서 기록된 40.877시간을 선형 환산하면 **77.2~90.8시간**이다.
새 hook overhead·실패 episode 증가에 따라 더 걸릴 수 있으며, 이 범위는 시간 상한이 아니다.
다운로드, 전체 시계열 재인코딩, 점수 계산, phase 보정, smoke/parity 실행은 포함하지 않는다.
GPU 대여 단가를 정한 뒤 비용을 계산하고, 작은 작동 검사의 실측 속도로 다시 추정한다.
기존 결과 재사용으로 줄일 경우에도 동일 초기상태·RNG·모델·SAE·환경·산술 및 raw action
일치 검증을 먼저 통과해야 한다. 호환성이 확인되지 않은 old/new 결과를 섞지 않는다.

### E3. 비교를 잘못 해석하지 않기 위한 규칙

1. **변경량:** episode 평균이 아니라 같은 hidden의 각 row에서 anchor delta 크기를 맞춘다.
   다른 SAE 방향은 자신의 활성값이 아니라 anchor의 활성 시점/크기를 빌린 방향 대조다.
   BF16 cast 전 크기와 cast 후 실제 크기를 모두 기록한다. closed-loop 궤적이 달라지면
   입력도 달라지므로 여러 조건의 누적 변경량이 같다고 주장하지 않는다.
   detector/threshold/rank를 결과를 본 뒤 유리하게 조정하지 않는다.
2. **강도:** `alpha`는 남기는 비율이다. `.75`는 25% 억제, `.5`는 50% 억제,
   `0`은 완전 제거다. 실제 성공률이 반드시 단조 감소한다고 가정하지 않는다.
3. **단계:** 단순 시간 3등분이나 baseline의 시간표를 현재 행동 단계라고 부르지 않는다.
   현재 pre-action 관측의 대상 물체 거리/실제 grasp/placement 정보가 필요하다.
   판별 규칙은 discovery/validation 영상과 대조하고, 평가 성공 여부로 튜닝하지 않는다.
   단계가 발생하지 않은 episode도 전체 성공률 분모에 남기고, 개입 노출 횟수를 별도 보고한다.
4. **action-token:** 올바른 token 집합을 policy/tokenizer에서 검증한다. subset 재정규화 KL은
   행동 token 밖으로 확률이 새는 현상을 숨길 수 있어 subset 확률 질량 변화도 함께 보고한다.
   이 점수도 fixed-prefix 지표이며, 물리적 행동 오차나 semantic interpretability 자체가 아니다.
5. **일반성:** Dr.VLA의 통계를 우리 readout SAE에 적용한 비교군이다. 전체 연속 시점의
   7개 latent를 encode 후 평균하며, 8개 시점 cache/Top64에서 없는 값을 0으로 메우지 않는다.
   논문 계수를 그대로 쓰거나 성공률로 일반성 라벨을 만들지 않는다. 자세한 원문/보정 규칙은 D2.

### E4. 오프라인 도구 — 입력 복원 후에만 사용

다음은 **새 데이터를 수집하거나 SAE를 학습하는 명령이 아니다.** 원본 dense 데이터와
기존 SAE·full10 config/split을 먼저 복원해야 한다. 현재 로컬에 전체 입력을 자동 다운로드하지 않는다.
`head-full10.yaml`의 기존 경로는 read-only로 읽고 결과 경로를 바꾸지 않는다.

계획만 확인 (`--execute`가 없으면 파일 접근/계산하지 않음):

```bash
python scripts/openvla/headfollowup.py temporal \
  --config configs/local/head-full10.yaml \
  --output /workspace/head-followup-v1/temporal
```

입력 검증·비용 확인 후 명시적으로 실행할 때:

```bash
python scripts/openvla/headfollowup.py temporal \
  --config configs/local/head-full10.yaml \
  --output /workspace/head-followup-v1/temporal \
  --device cuda:0 --execute

python scripts/openvla/headfollowup.py temporal-stats \
  --cache /workspace/head-followup-v1/temporal \
  --output /workspace/head-followup-v1/drvla-statistics.json
```

temporal output은 모든 시점의 **마지막 prediction readout**별 nonzero latent를 보존한다.
모든 image/prompt token의 latent를 보존했다는 뜻은 아니다. 한 source shard씩 mmap하고
latent를 작은 shard로 저장한다. 원본 content hash도 기록하지만, 과거 미기록 수집 상태가
이것으로 증명되는 것은 아니다. 중간 실패한 출력 폴더를 자동 덮어쓰거나 자동 재실행하지 않는다.
기존 Head cache와 겹치는 모든 readout의 nonzero ID·값이 재인코딩 결과와 정확히 일치하는지
확인한다. 다른 GPU/라이브러리 등으로 달라지면 원인을 확인할 때까지 멈추며 임계값을 몰래 완화하지 않는다.

`calibrate`는 실제 일반성 라벨이 준비된 후에만 사용한다. 라벨 JSON은
`drvla_discovery_labels_v1`, `split_role=discovery`, `used_outcomes=false`,
`used_head_rank=false`, `evaluator`, `statistics_sha256`, `labels`가 필요하다.
각 label은 `feature_id`, `label`(`general`/`episode_specific`), `evidence`를 갖는다.
15개씩 총 30개의 독립 라벨이 없으면 멈춘다. 테스트용 synthetic 라벨을 실제 라벨로 쓰지 않는다.

```bash
python scripts/openvla/headfollowup.py calibrate \
  --statistics /workspace/head-followup-v1/drvla-statistics.json \
  --labels /workspace/head-followup-v1/drvla-labels.json \
  --output /workspace/head-followup-v1/drvla-scores.json
```

①②③의 rollout 실행 명령은 아직 제공하지 않는다. phase adapter·확장 hook의 runner 연결·실제 GPU 검증이
완료된 뒤 새 조건표/예산/출력 디렉터리와 함께 별도로 만든다. 지금 원본 `run`을 재실행하지 않는다.

<a id="stage1-offline"></a>

### E5. 먼저 실행할 1단계 — ④·⑤만, 새 rollout 0회

**현재 상태:** 실제 백업에서 필요한 36개 파일(약 2.22GB)을 새 로컬 분석 폴더로 복원하고
해시 검증을 완료했다. pinned 모델 설정·tokenizer·코드로 action token 256개(ID 31744–31999)를
확인했다. 기존 6개 지표의 비교도 다시 계산했다. **새 action-KL/Dr.VLA 점수 계산 완료를 뜻하지 않는다.**
로컬에는 기존 SAE 및 원본 dense가 없고 GPU도 없다. 전체 dense는 약 281GiB라서 현재 로컬
여유 공간(복원 후 약 82GiB)으로 전체 다운로드하지 않는다.

후속 코드는 `research/drvla-followup` 브랜치에서 받는다. **기본 `main`을 clone하는 것만으로는
이 후속 CLI를 받을 수 없다.** 원격 브랜치 게시를 확인한 뒤 새 RunPod에서는 아래처럼 받는다.

```bash
cd /workspace
git clone --branch research/drvla-followup --single-branch https://github.com/jiyeon-yoon/Event-SAE-Head.git
cd Event-SAE-Head
git branch --show-current
git rev-parse HEAD
```

이미 같은 이름의 폴더가 있다면 삭제하거나 덮어쓰지 말고 저장소 상태부터 확인한다.
실행에 사용한 commit hash를 결과와 함께 기록한다. 9월 27일 개발 기록의 `commit/push=false`는
당시 상태를 보존한 것이며, 현재 원격 게시 상태는 Git으로 확인한다.
기존 결과를 원위치에 덮어쓰지 않는다. 여기서 `validate`는 저장된 입력에 대한 수치 검사이며
LIBERO 로봇 rollout이나 새로운 수집·학습이 아니다. 다른 GPU에서는 옛 인증서를 재사용하지 않는다.

#### 1. 백업·SAE·검증용 모델 복원

아래는 기존 RunPod runtime 환경, 저장 공간이 충분한 새 서버, 저장소 루트에서 실행하는 명령이다.
**실제 GPU에서 이 신규 전체 경로를 통과한 것은 아직 아니다.** 멈추면 gate를 우회하지 않는다.

```bash
STEP=/workspace/head-stage1-v1
CLI=scripts/openvla/headfollowup.py
mkdir -p "$STEP/mapping-source"

hf download jiyeony/event-sae-head-full10-results experiment.tar.gz manifest.json --repo-type dataset --revision 69c33d6a943c253dc97cf9a3968095276f7e9540 --local-dir "$STEP/bundle"

python "$CLI" stage1-restore --archive "$STEP/bundle/experiment.tar.gz" --manifest "$STEP/bundle/manifest.json" --output "$STEP/restored"
REF="$STEP/restored/results/full10-followup-v1"

hf download jiyeony/event-sae-openvla-libero-spatial-layer31-paper trainer_0/ae.pt trainer_0/config.json --revision adb776b08f5b8ec4ea67556cf3e3bc60d91d607a --local-dir "$STEP/sae"
hf download openvla/openvla-7b-finetuned-libero-spatial --revision 962318cec55ac10993ff0f5f43eda9a270b4c873 --local-dir /workspace/head-inputs/openvla-spatial
hf download openvla/openvla-7b configuration_prismatic.py modeling_prismatic.py processing_prismatic.py --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f
```

모델 전체 파일은 현재 환경의 **수치 검증** 때문에 필요하다. action scoring 자체는 복원된 head와
readout, SAE만 사용한다. 다운로드가 offline 환경 변수로 막혀 있으면 해당 다운로드 명령에만
`HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0`을 앞에 붙인다.

#### 2. action token 매핑과 현재 GPU 검증

```bash
hf download openvla/openvla-7b-finetuned-libero-spatial config.json tokenizer.json --revision 962318cec55ac10993ff0f5f43eda9a270b4c873 --local-dir "$STEP/mapping-source"
hf download openvla/openvla-7b modeling_prismatic.py --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f --local-dir "$STEP/mapping-source"

python "$CLI" action-map --source "$STEP/mapping-source" --head-manifest "$REF/head/output_head_manifest.json" --output "$STEP/action-mapping.json"

python "$CLI" stage1-setup --restored "$STEP/restored" --sae "$STEP/sae/trainer_0/ae.pt" --snapshot /workspace/head-inputs/openvla-spatial --output "$STEP/action-config"

python scripts/openvla/output_head_sensitivity.py validate --config "$STEP/action-config/action-runtime.yaml" --allow-model-execution
```

기존 token KL과 같은 `KL(original || feature_removed)` 방향이다. 비교 token 집합은 encoder가
표현하는 256개 token이며, native decoder가 모든 vocabulary ID를 clip하는 전체 정의역과는 다르다.
끝점 token 일부는 같은 bin center로 decode된다. **물리적 action 분포의 KL이라고 부르지 않는다.**

#### 3. ④ 계산 후 바로 비교

```bash
python "$CLI" action-score --config "$STEP/action-config/action-runtime.yaml" --reference-result "$REF" --mapping "$STEP/action-mapping.json" --output "$STEP/action-scores.json" --execute

python "$CLI" compare --reference-result "$REF" --action-scores "$STEP/action-scores.json" --output "$STEP/comparison-action"
```

기존 14,000개 readout 및 SAE 원래-forward latent를 그대로 쓴다. 100개 readout마다 진행률을 출력한다.
전체 vocabulary KL도 재계산해 이전 점수와의 최대 feature별 차이를 진단값으로 남긴다.
다른 장치의 수치 차이는 보고하며, 다른 산술의 결과를 bit-identical로 주장하지 않는다.
상위 feature가 기존 16개 평가 대상에 없으면 성공률 하락은 **미측정**이며 다른 feature로 대체하지 않는다.
실제 score 실행 시간은 측정 후 기록한다. 6,800~8,000회/77~91시간은 이 단계의 예산이 아니다.

#### 4. ⑤ 원본 재인코딩·통계

④는 원본 dense 다운로드 없이 먼저 끝낼 수 있다. ⑤를 진행할 때만 충분한 별도 저장 공간을 확보해
기존 pinned 원본을 복원·merge한다. 원본만 약 281GiB이며, 모델·archive·파생 cache 및 여유 공간이 추가로 필요하다.
이미 merged dense가 있다면 download/merge 두 명령은 생략한다. 이 명령들은 새 rollout/학습을 하지 않는다.

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python scripts/openvla/headfull.py download --execute
python scripts/openvla/headfull.py merge

DENSE=/workspace/event-sae-spatial-inputs/merged/sae_activations/post_mlp_residual
python "$CLI" stage1-setup --restored "$STEP/restored" --sae "$STEP/sae/trainer_0/ae.pt" --snapshot /workspace/head-inputs/openvla-spatial --dense "$DENSE" --output "$STEP/temporal-config"

python "$CLI" temporal --config "$STEP/temporal-config/temporal-input.yaml" --output "$STEP/temporal" --device cuda:0 --execute
python "$CLI" temporal-stats --cache "$STEP/temporal" --output "$STEP/drvla-statistics.json"
python "$CLI" label-packet --cache "$STEP/temporal" --statistics "$STEP/drvla-statistics.json" --output "$STEP/drvla-review.json"
```

원래 discovery 250개 episode의 **기록된 연속 시점 전체**가 대상이다. 기존 Head 시점과 겹치는
nonzero ID·값이 정확히 일치해야 한다. 누락 시점·차원, 기존 latent와 불일치하면 중단한다.
중간 실패 출력은 완성 cache로 취급하지 않는다. 8시점/Top64 cache를 완전한 시계열이라고 가정하지 않는다.

#### 5. 일반성 라벨은 사람의 검토 후 보정

`drvla-review.json`은 기존 평가 feature 16개를 제외한 후보 60개를 고정 seed로 선정하고,
모든 discovery episode의 전체 활성 시계열·영상 식별자를 제공한다. Head 순위·성공률은 넣지 않는다.
**영상 자체를 포함하지 않으므로** 원본 discovery 영상과 `source_run_id/task_id/task_episode_idx`를
대조해서 검토한다. 활성 coverage만으로 일반성을 자동 판정하지 않는다.

검토자는 `general`/`episode_specific`을 실제 의미·영상 근거로 판단하고, 불확실하면 제외한다.
양쪽 15개씩 30개가 안 모이면 임의로 채우지 않고 후보 검토를 확장한다. 라벨 형식은 E4와 같다.
`statistics_sha256`은 `sha256sum "$STEP/drvla-statistics.json"`으로 구한다.

```bash
python "$CLI" calibrate --statistics "$STEP/drvla-statistics.json" --labels "$STEP/drvla-labels.json" --output "$STEP/drvla-scores.json"
python "$CLI" compare --reference-result "$REF" --action-scores "$STEP/action-scores.json" --drvla-scores "$STEP/drvla-scores.json" --output "$STEP/comparison-stage1"
```

최종 산출물은 `action-scores.json`, 전체 temporal cache, `drvla-statistics.json`,
검토 자료·근거 라벨, `drvla-scores.json`, `comparison-stage1/{comparison.json,report.md}`다.
이 결과를 보고 ①②의 새 rollout을 할지 판단한다. **이 단계가 끝나도 새 후보의 행동 효과나
의미 있는 행동 feature라는 결론은 자동으로 생기지 않는다.**

## F. 개발 점검·이전 기록

```bash
git remote -v
git status --short
python -m pytest -q
git diff --check
```

현재 `origin`은 `jiyeon-yoon/Event-SAE-HeadRank`여야 한다. Head/Baseline/다른 팀원 저장소에는 push하지 않는다.
공용 설정 YAML의 입력은 비어 있으므로 `audit`이 경로 누락으로 blocked인 것은 정상이다.
GPU 결과 확인과 로컬 단위 테스트는 구분한다. macOS sandbox에서 OpenMP 공유 메모리 오류가
나면 환경 제약을 확인하며, 테스트 실패를 숨기거나 실험 수치를 바꾸지 않는다.

세부 CLI schema·과거 시행착오·SAE 학습/Baseline 재현은 [archive](archive)에 보관했다.
옛 문서 경로가 필요한 경우 문서 정리 전 commit `4a8dbf8`에서도 확인할 수 있다.

## G. Tail Full10 + 보정 예측기

현재 개발 branch는 `research/calibrated-head-tail`이다. 기존 main `183eaa9`와 기존 결과를
수정하지 않는다. 사용자가 승인한 게시 대상은 이 branch이며 main에는 병합하지 않는다.
**새 native 실행기의 실제 GPU 검증은 남아 있다.** 아래 명령은 이 branch의 게시 commit을
RunPod에서 받은 뒤 사용한다.
`prepare-run`은 추적된 소스가 깨끗한 commit이 아니면 중단한다. 결과 hash를 수동으로 바꾸지 않는다.

### G1. 규모와 해석

- 이미 완료한 활성 feature 6,009개의 `full-scores.json`을 재사용한다. 전체 scoring을 다시 하지 않는다.
- 행동 평가는 **새 feature 16개 + raw/identity 2조건**, 10 tasks × 20 cases = **3,600회**다.
- 기존 pilot/full10에서 결과를 본 22개와 팀원이 이미 평가한 ID를 제외한 뒤, Mean Head-KL·Tail
  Head-KL·Event-max 각각의 Top-3 합집합을 포함한다. 나머지는 고정된 순위 불일치/무작위 규칙으로 채운다.
- 여기서 Top-3는 **제외 후 후보군의 Top-3**다. 기존 전역 Top-3를 다시 평가한다는 뜻이 아니다.
- primary는 Tail, Mean/Event-max는 비교군이다. Excess는 기본적으로 Top-3 포함을 강제하지 않는다.
- 기존 3,600회는 약 40.9 condition-hour였다. 새 실행의 시간 보장이 아니며 준비·재시도 비용은 별도다.
- 현재 표본의 Mean/Tail 순위가 매우 비슷하므로 Tail의 개선을 기대하거나 보장하지 않는다. 이 실행은
  새 feature의 행동 효과를 측정하는 것이며, 새 지표의 우수성은 결과로 판단한다.

원본 scorer `183eaa9`의 정확히 허용된 구현과 그 출력만 호환된다. 점수 파일/score hash는 그대로
유지하며 현재 planner/runtime 구현을 따로 고정한다. 다른 오래된 출력까지 임의로 허용하지 않는다.
모델·SAE·캐시·장치·산술 조건 및 현재 GPU parity가 맞아야 한다.

### G2. Stop한 Pod 재시작 → GPU 테스트 → 팀 확인

RunPod 화면에서 **기존 Pod을 Start/Resume**하고 접속한다. Stop 이전 tmux/process는 종료됐으므로
새 세션을 만든다. Volume disk는 stop/start를 지나 유지되지만 container disk는 유실될 수 있다.
`/workspace`라는 이름만으로 영속 저장이 보장되지는 않으므로 마운트와 실제 파일을 확인한다.
출처: [RunPod storage](https://www.runpod.io/blog/where-did-my-files-go-a-straight-guide-to-runpod-storage).

```bash
event-sae-init
source /opt/conda/etc/profile.d/conda.sh
conda activate event-sae-openvla
event-sae-verify --require-gpu
df -h /workspace
tmux new -s headrank-tail
```

이후 모든 명령은 tmux 안에서 실행한다. 동일 이름의 세션이 있으면 새로 만들지 말고
`tmux attach -t headrank-tail`로 들어간다. BF16 연산까지 검증한다:

```bash
python -c 'import torch;x=torch.ones((128,128),device="cuda",dtype=torch.bfloat16);assert (x@x==128).all().item();print("BF16_CUDA_OK")'
cd /workspace/Event-SAE-HeadRank
git status --short
git fetch origin
git switch research/calibrated-head-tail
git pull --ff-only origin research/calibrated-head-tail
git rev-parse HEAD
git status --short
```

`git status --short`에 변경이 있거나 switch/pull이 실패하면 그 상태를 확인하고 멈춘다.
reset/clean/강제 덮어쓰기를 하지 않는다. local branch가 아직 없으면 fetch 후 switch가
origin의 동일 이름 branch를 추적해 생성한다. 실행 중에는 코드를 더 업데이트하지 않는다.

아래는 `/workspace/headrank-v1`에 기존 복원이 있는 Pod 기준이다. 새 Pod에서는 먼저 README의
복원·현재 장치 validation을 수행한다. 종료한 Pod의 shell 변수는 복구되지 않으므로 다시 지정한다.

```bash
cd /workspace/Event-SAE-HeadRank
STEP=/workspace/headrank-v1
REF="$STEP/restored/results/full10-followup-v1"
SPLIT="$STEP/restored/inputs/full10/split.json"
OLD_CFG="$STEP/config/action-runtime.yaml"
SCORES="$STEP/full-scores.json"
RUN=/workspace/headrank-tail-full10-v1
CFG="$RUN/config/action-runtime.yaml"
CLI=scripts/openvla/headrank.py
REG=configs/local/headrank-known.json
mkdir -p "$RUN"
```

이전 config에서 실제 SAE/model 경로를 읽어 확인한다. 하나라도 파일이 없으면 아래 GPU 실행으로
진행하지 않는다. Pod을 Stop했다는 이유만으로 281GiB 원본 자료를 다시 받지 않는다.

```bash
SAE=$(python -c 'import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))["inputs"]["sae_checkpoint"])' "$OLD_CFG")
SNAP=$(python -c 'import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))["inputs"]["local_model_snapshot"])' "$OLD_CFG")
ls -lh "$SCORES" "$SAE" "$SPLIT" "$OLD_CFG" "$SNAP/config.json"
ls -ld "$REF/readouts" "$REF/head"
```

이전 수치 검증 결과도 보존하도록 새 runtime 디렉터리를 만든다. 이미 `$RUN/config`가 있다면
재생성하지 말고 생성 이력을 먼저 확인한다.

```bash
python scripts/openvla/headfollowup.py stage1-setup \
  --restored "$STEP/restored" --sae "$SAE" --snapshot "$SNAP" \
  --output "$RUN/config"
python scripts/openvla/output_head_sensitivity.py validate \
  --config "$CFG" --allow-model-execution
```

정상 출력은 `status: passed`다. 이것은 모델을 실제 실행한 numerical parity 검증이며 로봇
성공률 실험은 아니다. 이어서 기존 개발 feature 16개만 작은 점수 계산으로 확인한다:

```bash
python "$CLI" score --config "$CFG" --reference-result "$REF" \
  --event-scores "$STEP/restored/inputs/full10/discovery_event_scores.pt" \
  --output "$RUN/gpu-smoke.json" --execute
```

기대값은 `status: scored`, `sampling_mode: smoke`, `features: 16`, `new_rollouts: 0`,
`development_only: true`, 두 `num_norm_invalid_*` 값 0이다. 이전 4090 smoke는 계산부 약62초였지만
로딩·I/O 시간은 별도다. **이 결과가 나온 지점에서 먼저 확인한다.** 기존 `full-scores.json`은
덮어쓰지 않고 본 실험에 그대로 사용한다. 새 smoke 결과로 full plan을 만들지 않는다.

새 descriptor 코드도 GPU 점검할 때는 E절처럼 action mapping을 검증해서 만든 뒤 G4의
`descriptors` 명령에 `--smoke`와 별도 output을 사용한다. 이 산출물은 학습/추론에는 못 쓴다.

Stop 중 container cache만 유실되어 `processing_prismatic.py`가 없다는 오류가 나면 아래
**고정 revision의 Python 파일 3개만** 받고 validate를 다시 실행한다:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf download openvla/openvla-7b \
  configuration_prismatic.py modeling_prismatic.py processing_prismatic.py \
  --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f
```

GPU·Torch/CUDA 버전 또는 snapshot 파일 metadata가 달라지면 prepare-run이 기존 점수 재사용을
차단할 수 있다. parity 통과와 score identity 일치는 별개다. 이때 hash나 허용오차를 수정해
통과시키지 말고 차이를 확인한다. 필요한 경우 점수 재계산 여부를 별도로 결정한다.

원본 full score SHA-256은
`aea94626851e0510b50085e4707b534d623867e12b8bf16068274f6d04fb2b4a`다.
백업은 HF dataset `jiyeony/event-sae-head-full10-results`, revision
`417c222030dab9c5c6e354a75ebb03f8057aa0b5`, 파일
`followups/headrank-20261011T104004Z/full-scores.json`에 있다. 새 다운로드 위치를 쓴다면
`SCORES`를 그 실제 경로로 지정한다.

registry가 없다면 예시를 복사하되 **기존 registry를 덮어쓰지 않는다**:

```bash
mkdir -p configs/local
test -e "$REG" || cp configs/research/openvla/headrank_known_features.example.json "$REG"
```

팀원에게 이미 개입 결과를 알고 있는 feature ID를 확인하고 `feature_ids`에 추가한다.
`reviewed_for_team_prior_outcomes`는 실제 확인 후에만 `true`로 설정한다. 현재 사용자의 답은
“팀원 확인 필요”이므로 이 확인이 남아 있다. 코드가 대신 확인 완료로 바꾸지 않는다.

### G3. 3,600회 계획 동결 — 아직 실행하지 않음

```bash
python "$CLI" budget --features 16 --tasks 10 --cases-per-task 20
python "$CLI" cases --split "$SPLIT" --cases-per-task 20 --output "$RUN/eval-cases.json"
python "$CLI" plan --scores "$SCORES" --registry "$REG" --eval-manifest "$RUN/eval-cases.json" --preset full10 --output "$RUN/tail-plan.json"
PLAN="$RUN/tail-plan.json"
python "$CLI" verify-plan --plan "$PLAN"
```

16개 ID, 각 지표의 Top-3 포함 여부, `total_rollouts: 3600`을 검토한다.
이미 있는 파일은 덮어쓰지 않는다. 설정을 바꿀 때는 새 출력 경로와 새 계획으로 만든다.

### G4. 선택 사항: 같은 16개에 보정 예측을 함께 검증

새 SAE를 학습하는 것이 아니다. 기존 16개 feature의 task별 성공률 하락을 이용해 **작은 ridge
회귀 예측기**를 개발한다. feature 하나의 모든 task를 함께 제외하는 LOFO 검증을 사용하고,
7024/24830을 함께 제외한 진단도 제공한다. 기존 자료의 편향 때문에 개발 검증이지 확증 결과는 아니다.
고정 alpha=1이며 개발 결과를 보고 값을 계속 바꾸면 별도 탐색으로 기록해야 한다.

가장 먼저 GPU 없이 기존 점수만으로 개발 진단을 만들 수 있다:

```bash
PCLI=scripts/openvla/headpredict.py
python "$PCLI" calibrate --scores "$SCORES" --reference-result "$REF" --output "$RUN/calibration-v1"
```

`report.md`에는 zero/KL/크기·빈도·Event/Head 비교군과 feature 단위 검증이 나온다.
좋은 결과를 가정하지 말고 이 결과부터 검토한다. **이 단계만 실행해도 로봇 rollout은 0회**다.

2026-10-11 실제 기존 자료의 CPU 개발 진단(추가 descriptor 없이)에서는 Head ridge의
feature-mean LOFO MAE가 0.1072, Spearman이 −0.2120이었다. KL ridge는 0.0920/0.3010이었다.
큰 손상 feature 7024/24830을 학습·평가에서 모두 제외하면 모든 비상수 예측기의 MAE가
단순한 “하락 0” 예측(0.0236)보다 나빴다. **현재 보정안의 우수성이 확인된 것이 아니며,
이 결과를 추가 GPU 지출의 근거로 삼지 않는다.** 이후 descriptor 기반 방법은 아직 미검증이다.

추가 출력 반응을 포함하려면 E절의 검증된 action mapping과 현재 GPU parity가 필요하다.
아래 `MAPPING`은 실제 생성한 경로로 바꾼다. 기존 16개만 원래 readout 전체에서 계산한다:

```bash
MAPPING="$STEP/action-mapping.json"
python "$PCLI" descriptors --config "$CFG" --reference-result "$REF" --mapping "$MAPPING" --output "$RUN/dev-descriptors.json" --execute
python "$PCLI" calibrate --scores "$SCORES" --reference-result "$REF" --descriptors "$RUN/dev-descriptors.json" --output "$RUN/response-calibration-v1"
```

descriptor에는 7개 생성 위치별 action-bin 기대값의 부호/절댓값 변화, conditional KL,
argmax 변화 등이 포함된다. 부호는 **좋다/나쁘다의 라벨이 아니며**, bin 변화는 물리적 action
오차가 아니다. 기존 8시점 표본에서 계산하므로 전체 연속 시계열이라고 부르지 않는다.
`--smoke` 산출물은 학습/추론 입력으로 받아들이지 않는다.

보정 예측을 이번 3,600회에 붙이려면 **새 rollout 전에** 예측값을 동결한다. 먼저 선택한
16개 ID를 확인한다:

```bash
python -c 'import json,sys;print(*json.load(open(sys.argv[1]))["panel_feature_ids"])' "$PLAN"
```

그 ID들을 다음 명령의 `--feature-ids` 뒤에 넣는다. 아래 `ID...`는 실제 정수 목록으로 바꿀
자리이며 그대로 실행할 수 있는 값이 아니다. 간단한 `head_ridge` 예측기는 추가 descriptor가 없다:

```bash
MODEL="$RUN/calibration-v1/model.json"
python "$PCLI" predict --model "$MODEL" --scores "$SCORES" --reference-result "$REF" --registry "$REG" --feature-ids ID... --output "$RUN/predictions.json"
python "$PCLI" attach --plan "$PLAN" --predictions "$RUN/predictions.json" --model "$MODEL" --output "$RUN/tail-plan-with-predictions.json"
PLAN="$RUN/tail-plan-with-predictions.json"
```

`response_ridge`를 선택한다면 위 16개에 대해서만 `descriptors --feature-ids ID...`로 새 파일을
만들고, response 모델과 `predict --predictor response_ridge --descriptors 해당파일`을 사용한다.
전 후보의 descriptor를 재계산할 필요는 없다. 학습·기존평가·팀 공개 ID가 섞이면 중단한다.
`attach`는 원본 계획을 수정하지 않고 예측/모델/hash가 포함된 새 계획을 만든다. 선택한 feature,
primary와 **3,600회 예산은 그대로**다. 보정 예측은 이 고정 패널의 보조 분석이며 전역 Top-K 성능은 아니다.

### G5. 준비, 소규모 control 확인, 재개

코드 변경을 마치고 승인된 clean commit에서 현재 GPU validation을 다시 수행한다.
이 문서는 자동 commit/push나 유료 실행의 승인이 아니다.

```bash
python scripts/openvla/output_head_sensitivity.py validate --config "$CFG" --allow-model-execution
python "$CLI" prepare-run --config "$CFG" --plan "$PLAN" --scores "$SCORES" --registry "$REG" --eval-manifest "$RUN/eval-cases.json" --reference-result "$REF" --split "$SPLIT" --output "$RUN/execution.json" --run-output "$RUN/runs" --max-rollouts 3600
python "$CLI" run --experiment "$RUN/execution.json"
```

여기까지는 validation 외에 로봇 rollout이 없다. 계획 내용/비용을 승인하고 tmux 안에서만 다음을 진행한다:

```bash
EXEC_HASH=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["execution_hash"])' "$RUN/execution.json")
CODE_REV=$(git rev-parse HEAD)
python "$CLI" run --experiment "$RUN/execution.json" --execute --approved-execution-hash "$EXEC_HASH" --expected-code-revision "$CODE_REV" --max-new-attempts 2
```

첫 raw/identity 1쌍 이후 `paused`는 정상이다. 이것만으로 제거 개입 검증을 마쳤다는 뜻은 아니다.
확인 후 같은 명령에서 `--max-new-attempts 2`를 빼면 재개한다. 완료 case는 건너뛰고 모든
200개 raw/identity 대조를 통과해야 제거 조건으로 진행한다. 재시도 reserve는 기본 0이며,
중단된 attempt가 있으면 자동으로 예산을 늘려 재시작하지 않는다. 실행 중 코드/계획을 바꾸지 않는다.

```bash
python "$CLI" run --experiment "$RUN/execution.json" --execute --approved-execution-hash "$EXEC_HASH" --expected-code-revision "$CODE_REV"
python "$CLI" status --experiment "$RUN/execution.json"
python "$CLI" analyze --experiment "$RUN/execution.json" --output "$RUN/analysis-v1"
```

분석은 모든 예정 case와 개입 증거가 있어야 완료된다. Tail/Mean/Event-max의 고정 패널 비교,
paired signed drop, Top-K와 uncertainty를 보고한다. 예측을 붙였다면 feature 평균 예측의 MAE와
Spearman도 별도로 보고한다. 보정 예측이 개선되었다고 자동 결론내리지 않는다.
이전 데이터의 초기상태/수집 flag 미확인, 기존 SAE/cluster 재사용, 이미 사용한 평가 라벨 이력은
이번 계획에서도 없어지지 않는다. 새 feature 평가와 완전히 새 task/환경 일반화는 다른 주장이다.
