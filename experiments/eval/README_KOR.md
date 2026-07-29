# experiments/eval

`peptide_experiment`에서 학습된 체크포인트(BOLT/ORPT/ORPT-FA/ORPT-LEX 등, 임의 milestone)를
비교 평가하는 파이프라인.

모든 스크립트는 아래 명령을 **BOLT 저장소 루트**에서 실행한다고 가정한다.

## TL;DR

준비물: 평가하고 싶은 (arm, milestone, run_dir, checkpoint_dir) 조합을 담은 manifest YAML 하나.
예시: [`manifests/poc20_four_arm.yaml`](manifests/poc20_four_arm.yaml) (BOLT/ORPT/ORPT-FA/ORPT-LEX ×
milestone 5/10/20).

아래를 그대로 복붙하면 raw proposal 생성 → 실험 1~3 compute → plot까지 한 번에 끝난다.
(`CONFIG`/`MANIFEST` 두 변수만 자기 것으로 바꾸면 됨. `CONFIG`는 모델을 지정하는 게 아니라, manifest의
모든 모델이 공유하는 상수 — `similarity_threshold`, `oracle_budget`, `table_k_checkpoints`, task
universe(trainset/heldout 정의) — 를 읽기 위한 것일 뿐이므로, manifest에 속한 모델 중 아무 config나
(그 필드값들만 같다면) 넣으면 된다 — 자세한 이유는 "0. 준비물"과 실험 1 아래 참고.)

```bash
CONFIG=peptide_experiment/configs/peptide_poc20_bolt.yaml
MANIFEST=experiments/eval/manifests/poc20_four_arm.yaml
RESULTS=experiments/eval/results/poc20_four_arm

# 0) raw proposal 생성 (manifest의 모든 arm/milestone 대상, 한 번만 실행하면 됨)
python experiments/eval/generate_raw_proposals.py --config $CONFIG --manifest $MANIFEST

# 1) incumbent vs pool size (BO 없음, 저렴함 -- 여기서부터 확인)
python experiments/eval/incumbent_vs_pool_size.py --config $CONFIG --manifest $MANIFEST
python experiments/eval/plot_incumbent_vs_pool_size.py --results-dir $RESULTS

# 2) fixed-target pool BO (실제 LOLBO 실행 -- 비용 큼)
# sanity check: python experiments/eval/fixed_target_rejection_bo.py --config $CONFIG --manifest $MANIFEST --limit-tasks 1 --target-pool-sizes 10
python experiments/eval/fixed_target_rejection_bo.py --config $CONFIG --manifest $MANIFEST
python experiments/eval/plot_fixed_target_rejection_bo.py --results-dir $RESULTS

# 3) fixed-budget rejection-sampled BO (실제 LOLBO 실행 -- 비용 큼)
# sanity check: python experiments/eval/fixed_budget_rejection_bo.py --config $CONFIG --manifest $MANIFEST --limit-tasks 1
python experiments/eval/fixed_budget_rejection_bo.py --config $CONFIG --manifest $MANIFEST
python experiments/eval/plot_fixed_budget_rejection_bo.py --results-dir $RESULTS
```

결과 PNG는 `$RESULTS/plots/`에 쌓인다. 각 단계가 정확히 무엇을 하는지, input/output이 뭔지는
아래 실험별 설명 참고.

## 0. 준비물: manifest

평가하고 싶은 (arm, milestone, run_dir) 조합을 담은 YAML 파일 하나가 필요하다.
예시: [`manifests/poc20_four_arm.yaml`](manifests/poc20_four_arm.yaml)

```yaml
models:
  - arm: BOLT
    milestone: 5
    run_dir: runs/peptide_poc20_bolt
    checkpoint_dir: runs/peptide_poc20_bolt/checkpoints/BOLT-5/epoch_4  # generate_raw_proposals.py만 사용
  - arm: ORPT
    milestone: 5
    run_dir: runs/peptide_poc20_orpt
    checkpoint_dir: runs/peptide_poc20_orpt/checkpoints/ORPT-5/epoch_0
  # ...
```

- 경로는 저장소 루트 기준 상대경로.
- `checkpoint_dir`는 해당 (arm, milestone)의 raw proposal이 이미 존재하면 생략 가능.
- 서로 다른 `experiment_id`/config로 학습된 모델도 한 manifest에 자유롭게 섞을 수 있다.

그리고 모든 분석 스크립트는 `--config <experiment yaml>` 하나를 추가로 받는다. 이 config는
모델을 지정하는 게 아니라, manifest 전체가 공유하는 상수(`similarity_threshold`,
`oracle_budget`, `table_k_checkpoints`, task universe)만 읽기 위한 것이므로, manifest에 속한
모델들의 config 중 아무거나(그 필드들만 같으면) 쓰면 된다.

Task universe는 디스크에서 발견하는 게 아니라 고정되어 있다: `trainset` = 비교 대상 milestone들이
전부 이미 학습한 task(`range(min(cfg.milestones))`), `heldout` = `cfg.heldout_tasks("heldout20")`.

## 1. 공통 선행 단계: raw proposal 생성

어떤 실험(1~3)을 돌리든, 그 전에 딱 한 번 실행해야 하는 단계.

| | |
|---|---|
| 스크립트 | [`generate_raw_proposals.py`](generate_raw_proposals.py) |
| 하는 일 | **`--manifest`에 들어있는 모든 (arm, milestone) 항목 전부**에 대해, 각자의 `checkpoint_dir`에서 체크포인트를 로드하고, 고정된 task 집합(`trainset`/`heldout`)마다 raw candidate peptide를 샘플링 |
| Input | `--config`, `--manifest` |
| Output | `<run_dir>/eval_raw/<task_set>/<arm>-<milestone>/task_<idx>_sampled_attempt*.jsonl` |

```bash
python experiments/eval/generate_raw_proposals.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml
```

`--config`와 `--manifest`의 역할이 다르다는 점에 주의:
- **평가 대상 모델(누구를 위한 raw proposal인지)을 정하는 건 `--manifest`뿐이다.** 위 명령은
  `poc20_four_arm.yaml`에 나열된 12개 항목(BOLT/ORPT/ORPT-FA/ORPT-LEX × milestone 5/10/20) 전부에
  대해 raw proposal을 생성한다 — BOLT만 생성하는 게 아니다.
- **`--config`는 모델을 지정하지 않는다.** manifest의 모든 항목이 공유하는 상수
  (`similarity_threshold`, task universe 등)를 읽기 위한 용도일 뿐이라, 예시처럼
  `peptide_poc20_bolt.yaml`을 넘겨도 ORPT/ORPT-FA/ORPT-LEX 체크포인트 역시 정상적으로 처리된다.
  (단, 이 4-arm PoC config들처럼 해당 필드값이 서로 같아야 함 — 앞의 "0. 준비물" 참고.)

manifest에 있는 (arm, milestone, task_set) 조합의 raw proposal이 이미 존재하면 이 스크립트를
다시 돌릴 필요 없음. 아래 실험 1~3은 전부 이 결과물 위에서 동작하는 순수 post-hoc 분석이며,
`checkpoint_dir`나 LLM 샘플링을 다시 건드리지 않는다.

---

## 실험 1: Proposal-level incumbent vs. pool size (BO 없음)

"모델이 몇 번째 feasible proposal까지 뽑았을 때 최고 점수가 얼마나 좋아지는가"를 저렴하게 보는
지표. BO를 돌리지 않으므로 여기서부터 시작하는 것을 권장.

| | |
|---|---|
| Compute | [`incumbent_vs_pool_size.py`](incumbent_vs_pool_size.py) |
| Plot | [`plot_incumbent_vs_pool_size.py`](plot_incumbent_vs_pool_size.py) |

**Compute**
- Input: 실험 0에서 만든 raw proposal (`<run_dir>/eval_raw/...`), `--config`, `--manifest`
- 하는 일: 각 (arm, milestone, task_set, task_idx)마다 raw jsonl들을 읽어 feasible/unique
  proposal pool을 만들고, `apex_wrapper`로 점수를 새로 계산. `n_proposals` 체크포인트
  (기본 `[1, 5, 10, 20, 50]`)마다 (a) 그 개수까지의 incumbent(최저 MIC), (b) 그만큼 뽑는 데
  필요했던 raw draw 수/rejection rate를 기록
- Output: `results/<manifest stem>/per_task_incumbent_vs_pool_size.csv`,
  `results/<manifest stem>/summary_incumbent_vs_pool_size.csv`

```bash
python experiments/eval/incumbent_vs_pool_size.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml
```

**Plot**
- Input: `--results-dir results/<manifest stem>`
- 하는 일: task_set(trainset/heldout)별로 두 종류의 그림을 그림
  - `incumbent_mic_bymilestone_n<n_proposals>_<task_set>.png`: 고정된 `n_proposals`(기본 10)에서
    arm별 milestone 추이 (헤드라인 차트)
  - `incumbent_mic_byNProposals_<task_set>.png`: `n_proposals` 체크포인트별 small multiples
- Output: `results/<manifest stem>/plots/*.png`

```bash
python experiments/eval/plot_incumbent_vs_pool_size.py \
    --results-dir experiments/eval/results/poc20_four_arm
```

---

## 실험 2: Fixed-target pool BO

모든 task를 **같은 크기**의 feasible pool(target_pool_size)로 강제 정렬한 뒤 실제 BO를 돌려서,
초기화 조건을 동일하게 맞춘 상태로 비교. 실제 LOLBO 실행이 들어가므로 비용이 크다 —
`--limit-tasks`로 먼저 사인체크할 것.

| | |
|---|---|
| Compute | [`fixed_target_rejection_bo.py`](fixed_target_rejection_bo.py) |
| Plot | [`plot_fixed_target_rejection_bo.py`](plot_fixed_target_rejection_bo.py) |

**Compute**
- Input: raw proposal, `--config`, `--manifest`, `--target-pool-sizes`(기본 `10,20,50`), `--limit-tasks`(선택)
- 하는 일: 각 task에서 raw proposal로부터 정확히 `target` 개의 feasible/unique pool을 rejection
  sampling으로 만들 수 있으면(추가 샘플링 없음) 그 pool로 실제 BO(LOLBO)를 실행. 목표 크기를
  못 채우는 task는 스킵하고 `coverage_rate`로 별도 집계(평균에 묻지 않음)
- Output: `results/<manifest stem>/fixed_target_bo_coverage.csv`,
  `results/<manifest stem>/per_task_fixed_target_bo.csv`,
  `results/<manifest stem>/summary_fixed_target_bo.csv`
  (BO 실행 중 생성되는 무거운 per-task artifact는 `<run_dir>/eval_fixed_target_bo/`에 저장)

```bash
# sanity check: task 1개, target 1개
python experiments/eval/fixed_target_rejection_bo.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml \
    --limit-tasks 1 --target-pool-sizes 10

# 전체 실행
python experiments/eval/fixed_target_rejection_bo.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml
```

**Plot**
- Input: `--results-dir results/<manifest stem>`
- 하는 일: 세 종류의 그림
  - `fixedtarget_mic_bymilestone_target<T>_bo<bo_calls>_<task_set>.png`: 기준
    target_pool_size(기본: 존재하는 것 중 최대)와 bo_calls(기본 5000)에서 헤드라인 차트
  - `fixedtarget_mic_bytarget_bo<bo_calls>_<task_set>.png`: target_pool_size별 small multiples
  - `fixedtarget_rejection_bymilestone_target<T>_<task_set>.png`: 기준 target에서 rejection rate
    (raw draw 대비 몇 %를 버렸는지) vs. milestone
- Output: `results/<manifest stem>/plots/*.png`

```bash
python experiments/eval/plot_fixed_target_rejection_bo.py \
    --results-dir experiments/eval/results/poc20_four_arm
```

---

## 실험 3: Fixed-budget (real) rejection-sampled BO

목표 pool 크기를 강제하지 않고, 고정된 샘플링 예산(`generate_raw_proposals.py`가 만든 만큼) 안에서
실제로 살아남는 feasible pool 그대로 BO를 돌림. Pool 크기가 task/arm마다 달라지는 것 자체가
측정 대상(제약 위반이 잦은 arm일수록 pool이 작아짐).

| | |
|---|---|
| Compute | [`fixed_budget_rejection_bo.py`](fixed_budget_rejection_bo.py) |
| Plot | [`plot_fixed_budget_rejection_bo.py`](plot_fixed_budget_rejection_bo.py) |

**Compute**
- Input: raw proposal, `--config`, `--manifest`, `--limit-tasks`(선택), `--min-feasible`(기본 5, LOLBO
  trust-region hang을 피하기 위한 최소 pool 크기 하한)
- 하는 일: 각 task의 raw proposal에서 뽑히는 feasible pool(크기 가변) 그대로 BO 실행. 하한을
  못 채운 task는 스킵하고 `coverage_rate`/pool 크기 분포로 별도 집계
- Output: `results/<manifest stem>/fixed_budget_bo_coverage.csv`,
  `results/<manifest stem>/per_task_fixed_budget_bo.csv`,
  `results/<manifest stem>/summary_fixed_budget_bo.csv`
  (무거운 per-task artifact는 `<run_dir>/eval_fixed_budget_bo/`에 저장)

```bash
# sanity check
python experiments/eval/fixed_budget_rejection_bo.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml \
    --limit-tasks 1

# 전체 실행
python experiments/eval/fixed_budget_rejection_bo.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml
```

**Plot**
- Input: `--results-dir results/<manifest stem>`
- 하는 일: 세 종류의 그림
  - `fixedbudget_mic_bymilestone_bo<bo_calls>_<task_set>.png`: 기준 bo_calls(기본 5000, 즉 전체
    oracle budget 소진 시점)에서 헤드라인 차트
  - `fixedbudget_mic_byboCalls_<task_set>.png`: bo_calls 체크포인트별 small multiples
  - `fixedbudget_rejection_bymilestone_<task_set>.png`: rejection rate vs. milestone
- Output: `results/<manifest stem>/plots/*.png`

```bash
python experiments/eval/plot_fixed_budget_rejection_bo.py \
    --results-dir experiments/eval/results/poc20_four_arm
```

---

## 여러 결과셋을 한 그래프에 합쳐 보기

모든 `plot_*.py`는 `--results-dir`에 콤마로 구분된 여러 경로를 받아 concat 후 그린다. 서로 다른
manifest를 따로 실행했더라도 한 그림에 모아 비교할 수 있다.

```bash
python experiments/eval/plot_incumbent_vs_pool_size.py \
    --results-dir experiments/eval/results/poc20_four_arm,experiments/eval/results/other_manifest
```

## 참고

- `results/`는 gitignore 대상 (`experiments/*/results/` glob으로 이미 커버됨).
- 파일 구조:

```
experiments/eval/
├── manifests/                        # 비교 대상 (arm, milestone, run_dir) 정의
├── generate_raw_proposals.py         # 0단계: raw proposal 생성 (유일하게 checkpoint를 읽음)
├── incumbent_vs_pool_size.py         # 실험 1 compute
├── plot_incumbent_vs_pool_size.py    # 실험 1 plot
├── fixed_target_rejection_bo.py      # 실험 2 compute
├── plot_fixed_target_rejection_bo.py # 실험 2 plot
├── fixed_budget_rejection_bo.py      # 실험 3 compute
├── plot_fixed_budget_rejection_bo.py # 실험 3 plot
├── common.py / plot_common.py        # 공용 유틸 (manifest 로딩, pool 구성, 스타일 등)
└── results/                          # output CSV/PNG, manifest stem별 하위 디렉토리 (gitignored)
```
