# DatawiseAgent q_review and Voting Extension

This repository contains my course-project modifications to the original DatawiseAgent codebase. The original upstream README has been preserved as [original.md](original.md). This README focuses on what I changed, how the new evaluation workflow works, and what effect the changes had on InfiAgent-Bench.

## Overview

The original DatawiseAgent evaluation workflow runs one agent trajectory per benchmark question: create a session, upload the table, let the agent plan, write code, execute code, self-debug, and return the final answer. For InfiAgent-Bench, this single-trajectory workflow is fragile because:

- A single run can fail because of randomness, column-name mistakes, format errors, or failed self-debugging.
- The final answer can look plausible even when the executed evidence in the notebook trajectory is incomplete.
- InfiAgent-Bench has 257 questions, and one question can trigger many LLM calls and code executions.
- Long-running benchmark jobs can leave Jupyter/Docker resources alive if a trajectory times out or crashes.

I added three main components:

- `q_review`: a final, non-corrective review stage that audits a completed trajectory.
- Majority vote and review-aware vote: multi-trajectory aggregation for self-consistency.
- Runtime protection: per-trajectory HTTP timeout and explicit Jupyter/Docker cleanup.

## Main Changes

### q_review

Related files:

- `datawiseagent/prompts/datawise.py`
- `datawiseagent/common/types/server.py`
- `datawiseagent/common/types/node.py`
- `datawiseagent/agents/datawise_agent.py`
- `evaluation/eval_infiagent_bench.py`
- `evaluation/review_aware_vote.py`

I added a new agent configuration field:

```yaml
agent:
  review:
    enabled: False
```

It is disabled by default to keep the baseline compatible with the original single-trajectory behavior. The InfiAgent-Bench evaluation script enables it automatically for:

- `q_review`
- `review_aware_vote`

It stays disabled for:

- `baseline`
- `mv`

In `DatawiseAgent.chat()`, I added a new state named `REVIEW`. In the original workflow, the agent terminated immediately after emitting `<Fulfill USER INSTRUCTION>`. With review enabled, the agent enters `_review_trajectory()` before returning the final response.

The review stage is intentionally a detector, not a corrector:

- It reads the full trajectory, including the user query, code cells, outputs, debugging trace, and final answer.
- It emits exactly one verdict: `<No_Issues>` or `<Issues_Found>`.
- It does not generate repair code.
- It does not rerun experiments.
- It does not transition back to execution or debugging.
- It appends a `ReviewNode` to the notebook history for auditing and aggregation.

The review prompt checks for common failure modes:

- Data leakage.
- Metric or statistic mismatch.
- Incorrect split, sampling, filtering, or grouping.
- Column-name, dtype, missing-value, or unit mistakes.
- Randomness and reproducibility issues.
- Invalid file paths or outputs.
- Final answers unsupported by executed evidence.
- Required answer-format violations.

The verdict parser is conservative. If both verdict tokens appear, or if the verdict is ambiguous or missing, the parser returns `<Issues_Found>`. This prevents malformed review text from being treated as clean.

### Majority Vote

Related files:

- `evaluation/review_aware_vote.py`
- `evaluation/eval_infiagent_bench.py`

I added a `TrajectoryResult` record for each run:

```python
index: int
user_id: str
session_id: str
verdict: str
answer: str
response_content: str
```

Each result JSONL record keeps:

- `response`: the selected answer used for official reformat/evaluation.
- `selected_index`: the selected trajectory index.
- `selected_verdict`: the review verdict for the selected trajectory.
- `trajectories`: all raw trajectories for auditing.

The vote does not compare full raw responses. Instead, `extract_answer()` extracts the most likely final-answer block from the trajectory, and `answer_signature()` canonicalizes it before voting:

- If the answer contains `@name[value]`, those formatted pairs are used as the vote signature.
- Otherwise, metric-value pairs are extracted when possible.
- Otherwise, numeric values and simple categorical labels are extracted.
- If none of those work, normalized text is used.

`majority_vote()` groups trajectories by signature. If there is a unique largest group, it selects the earliest trajectory in that group. If the vote is tied, it returns `None`, and the evaluation script falls back to the first trajectory.

### Review-Aware Vote

`review_aware_aggregate()` combines q_review with majority vote:

1. Select trajectories whose verdict is `<No_Issues>`.
2. If any clean trajectories exist, run majority vote only over the clean subset.
3. If no clean trajectories exist, run majority vote over all trajectories.
4. If the vote ties, fall back to the first trajectory.

The goal is to use majority vote to reduce sampling noise while using q_review to downweight trajectories that show execution or reasoning risks.

### Evaluation Script Updates

`evaluation/eval_infiagent_bench.py` now supports:

```bash
--method baseline|q_review|mv|review_aware_vote
--trajectories 3
--vote_temperature 0.7
--trajectory_temperatures 0,0.7,1.0
--seed_results_path <existing-q-review-jsonl>
--derived_mv_result_path <write-derived-mv-jsonl>
--chat_timeout <seconds>
```

Important options:

- `--seed_results_path`: reuse an existing q_review or review-aware result as trajectory 1.
- `--derived_mv_result_path`: while running review-aware vote, also write ordinary majority-vote results from the same trajectories.
- `--trajectory_temperatures`: use different temperatures across trajectories to increase diversity.
- `--chat_timeout`: cap one agent chat trajectory.

## Runtime Protection

### Per-Trajectory Timeout

Related files:

- `evaluation/eval_infiagent_bench.py`
- `evaluation/review_aware_vote.py`

I added `--chat_timeout` to the InfiAgent-Bench runner. It controls the HTTP timeout for one `/chat/` request.

The current default in `eval_infiagent_bench.py` is `900` seconds. For a strict 10-minute limit, pass:

```bash
--chat_timeout 600
```

Example:

```bash
python evaluation/eval_infiagent_bench.py \
  --method baseline \
  --note baseline-gpt4o-mini-full \
  --chat_timeout 600 \
  --num_workers 1
```

The timeout path is:

1. `eval_infiagent_bench.py` passes `args.chat_timeout` to `run_trajectory()`.
2. `run_trajectory()` calls `chat(..., timeout=chat_timeout)`.
3. `review_aware_vote.py::chat()` calls `httpx.post("/chat/", timeout=timeout)`.
4. If the trajectory exceeds the timeout, `httpx` raises an exception.
5. `run_trajectory()` enters `finally` and calls `stop_session(user_id, session_id)`.
6. The failed question is not written to the result file, so future runs can resume from completed IDs.

This timeout applies to one trajectory, not necessarily one whole benchmark question. For `review_aware_vote --trajectories 3`, one question may run up to three trajectories.

### Docker and Jupyter Cleanup

Related files:

- `main.py`
- `evaluation/review_aware_vote.py`
- `evaluation/eval_infiagent_bench.py`
- `datawiseagent/agents/datawise_agent.py`
- `datawiseagent/coding/jupyter/jupyter_code_executor.py`
- `datawiseagent/coding/jupyter/docker_jupyter_server.py`
- `datawiseagent/coding/jupyter/start.sh`

I added a backend endpoint:

```text
POST /users/{user_id}/sessions/{session_id}/stop
```

Cleanup flow:

1. `evaluation/review_aware_vote.py::stop_session()` sends the stop request.
2. `main.py::stop_session()` calls `agent.stop_session(session_id)`.
3. `DatawiseAgent.stop_session()` removes the session from memory and calls `code_executor.stop()`.
4. `JupyterCodeExecutor.stop()` best-effort stops the kernel, deletes the kernel, and stops the Jupyter server.
5. If Docker is used, `DockerJupyterServer.stop()` runs the container cleanup function.

`run_trajectory()` calls `stop_session()` inside `finally`, so cleanup is attempted after normal completion, timeout, or exception.

Docker sessions are created with:

```python
DockerJupyterServer(
    auto_remove=True,
    stop_container=True,
    ...
)
```

This means:

- `stop_container=True`: register a cleanup callback to stop the container.
- `auto_remove=True`: remove the container automatically after it stops.

The container-side startup script `datawiseagent/coding/jupyter/start.sh` also checks whether port 8888 is already occupied inside the container and kills the occupying process before starting Jupyter Kernel Gateway.

## Experimental Results

The following results are from the InfiAgent-Bench development set with 257 questions. Metrics were computed with the official `eval_closed_form.py` after reformatting the model responses.

| Method | Question Accuracy | Sub-Question Accuracy | Proportional Sub-Question Accuracy | Correct Questions |
| --- | ---: | ---: | ---: | ---: |
| baseline | 71.98% | 74.78% | 77.43% | 185/257 |
| q_review | 73.15% | 76.10% | 78.11% | 188/257 |
| majority vote | 73.15% | 76.10% | 77.79% | 188/257 |
| review-aware vote | 73.93% | 76.32% | 78.31% | 190/257 |

Compared with baseline:

- `q_review` improves question accuracy by 1.17 points, with a net gain of 3 correct questions.
- Majority vote also improves question accuracy by 1.17 points, with a net gain of 3 correct questions.
- Review-aware vote improves question accuracy by 1.95 points, with a net gain of 5 correct questions.

The q_review signal is conservative. In the full q_review run, only 11 of 257 trajectories were marked `<No_Issues>`, while 246 were marked `<Issues_Found>`. This is expected from the prompt design: the review stage is intended to flag uncertainty rather than generously clear trajectories.

Review-aware vote performs best among the tested variants, but the improvement is modest relative to its cost. It runs three trajectories by default and adds review calls, so it is substantially more expensive than baseline.

Concept-level observations:

- Summary Statistics improved from 76.67% to 83.33% under q_review/review-aware vote.
- Comprehensive Data Preprocessing improved from 44.44% to 57.78%.
- Outlier Detection improved from 65.71% to 71.43% under review-aware vote.
- Correlation Analysis decreased from 72.22% to 69.44% under review-aware vote, so the method is not uniformly better across all concept types.

## Reproduction Commands

Start the backend:

```bash
cd /mnt/data/fulin/510_bighw/DatawiseAgent
python main.py
```

Run baseline with a 10-minute per-trajectory timeout:

```bash
python evaluation/eval_infiagent_bench.py \
  --method baseline \
  --note baseline-gpt4o-mini-full \
  --chat_timeout 600 \
  --num_workers 1
```

Run q_review:

```bash
python evaluation/eval_infiagent_bench.py \
  --method q_review \
  --note q_review-gpt4o-mini-full \
  --chat_timeout 600 \
  --num_workers 1
```

Run review-aware vote and also export ordinary majority vote from the same trajectories:

```bash
python evaluation/eval_infiagent_bench.py \
  --method review_aware_vote \
  --note review_aware_vote-gpt4o-mini-full \
  --trajectories 3 \
  --trajectory_temperatures 0,0.7,1.0 \
  --seed_results_path evaluation/experimental_results/InfiAgent-Bench/results_q_review_gpt4o-mini-full.jsonl \
  --derived_mv_result_path evaluation/experimental_results/InfiAgent-Bench/results_mv_from_mvr_gpt4o-mini-full.jsonl \
  --chat_timeout 600 \
  --num_workers 1
```

Reformat answers:

```bash
cd evaluation
python InfiAgentBench/scripts/reformat.py \
  --model gpt-4o-mini \
  --responses_file_path experimental_results/InfiAgent-Bench/results_review_aware_vote_gpt4o-mini-full.jsonl \
  --output_file_path experimental_results/InfiAgent-Bench/reformat/results_reformat_review_aware_vote-gpt4o-mini-full.jsonl
```

`reformat.py` writes formatted answers to `reformat_response`, but the official evaluator reads `response`. Before evaluation, create an eval version where `response` is replaced by `reformat_response`:

```bash
python - <<'PY'
import json
from pathlib import Path

src = Path("experimental_results/InfiAgent-Bench/reformat/results_reformat_review_aware_vote-gpt4o-mini-full.jsonl")
dst = Path("experimental_results/InfiAgent-Bench/reformat/results_eval_review_aware_vote-gpt4o-mini-full.jsonl")
with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        item = json.loads(line)
        item["raw_response"] = item.get("response", "")
        item["response"] = item.get("reformat_response", item.get("response", ""))
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
PY
```

Evaluate closed-form answers:

```bash
cd evaluation
python InfiAgentBench/scripts/eval_closed_form.py \
  --questions_file_path InfiAgentBench/data/da-dev-questions.jsonl \
  --labels_file_path InfiAgentBench/data/da-dev-labels.jsonl \
  --responses_file_path experimental_results/InfiAgent-Bench/reformat/results_eval_review_aware_vote-gpt4o-mini-full.jsonl
```

## Summary

The added q_review stage provides a conservative quality signal for completed trajectories. Majority vote improves robustness through self-consistency. Review-aware vote combines both and achieved the best result in my InfiAgent-Bench run: 73.93% question accuracy versus 71.98% for baseline.

The gain is modest, but the workflow is useful for analyzing trajectory reliability. The timeout and Docker/Jupyter cleanup changes are also necessary for running the full benchmark without accumulating stalled sessions or leftover containers.
