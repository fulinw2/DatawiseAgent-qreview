# DatawiseAgent 修改说明与实验分析

本文档记录当前仓库相对于原始 DatawiseAgent 代码的主要修改，重点包括 InfiAgent-Bench 上加入的 `q_review`、majority vote / review-aware vote 机制，以及为长时间跑 benchmark 增加的单题超时和 Docker/Jupyter 资源清理链路。

## 修改目标

原始 DatawiseAgent 在 InfiAgent-Bench 上主要按单条轨迹运行：每道题创建一个 session，上传对应表格，让 agent 规划、写代码、执行、debug，最后直接取最终回答。这个流程的问题是：

- 单次轨迹容易受随机性、列名误判、格式错误、debug 失败影响。
- Agent 生成的是完整 notebook 轨迹，最终答案可能看起来合理，但中间执行证据不足。
- InfiAgent-Bench 一共 257 道题，单题可能包含几十次 LLM 调用和多次代码执行，跑完整 benchmark 容易出现超时或遗留 Docker 容器。

因此我在原有流程上增加了三类能力：

- `q_review`：在 agent 完成答案后增加一次只审查、不修复的最终质量审查。
- majority vote / review-aware vote：多条轨迹自一致投票，并可结合 `q_review` 的 verdict 过滤低可信轨迹。
- 运行保护：为单题设置 HTTP chat timeout，并在每条 trajectory 结束或异常时主动停止 session、清理 Jupyter/Docker 资源。

## q_review 修改

相关文件：

- `datawiseagent/prompts/datawise.py`
- `datawiseagent/common/types/server.py`
- `datawiseagent/common/types/node.py`
- `datawiseagent/agents/datawise_agent.py`
- `evaluation/eval_infiagent_bench.py`
- `evaluation/review_aware_vote.py`

### 配置入口

在 agent config 中新增 `review.enabled`：

```yaml
agent:
  review:
    enabled: False
```

默认关闭，保证 baseline 行为和原始单轨迹流程兼容。评测脚本会根据 method 自动打开：

- `baseline`：不开启 review。
- `q_review`：开启单条轨迹 review。
- `mv`：不开启 review，只做普通 majority vote。
- `review_aware_vote`：开启 review，并用 verdict 参与聚合。

### Agent 状态机改动

在 `DatawiseAgent.chat()` 的规划状态机里新增 `REVIEW = "q_review"` 状态。原始流程在 `<Fulfill USER INSTRUCTION>` 后直接结束；修改后，如果 `session.agent_config.review.enabled=True`，会进入 `_review_trajectory()`。

`q_review` 的设计原则是 detector，不是 corrector：

- 只阅读完整轨迹，包括用户问题、代码、执行输出、debug 过程和最终答案。
- 输出 verdict：`<No_Issues>` 或 `<Issues_Found>`。
- 不允许写修复代码、不允许重新运行实验、不允许回到 debug 阶段。
- 将 review 结果作为 `ReviewNode` 追加到 history，方便后续 audit 和聚合。

### Prompt 设计

`REVIEW_STAGE_SYSTEM_PROMPT` 和 `REVIEW_APPEND_PROMPT` 增加了一个保守审查清单：

- 数据泄漏。
- metric / statistic 是否和题目一致。
- split、sampling、filter、grouping 是否正确。
- 列名、dtype、缺失值、单位是否处理正确。
- 随机性和可复现性。
- 文件读写路径是否正确。
- 最终答案是否由执行证据支持。
- 输出格式是否满足题目要求。

策略上故意保守：证据不充分、格式不稳、执行链条不完整时默认 `<Issues_Found>`，避免把可疑轨迹当成高可信轨迹。

### Verdict 解析

`_parse_review_verdict()` 对 verdict 做保守解析：

- 优先识别 `Review Verdict: <No_Issues>` / `Review Verdict: <Issues_Found>`。
- 如果两个 verdict 都出现，按 `<Issues_Found>` 处理。
- 如果只在文本里零散提到一个 verdict，也可以解析。
- 如果缺失或歧义，默认 `<Issues_Found>`。

这个策略避免模型在 review 文本里同时提到两个标签时误判为 clean。

## Majority Vote 与 Review-Aware Vote

相关文件：

- `evaluation/review_aware_vote.py`
- `evaluation/eval_infiagent_bench.py`

### 轨迹结构

新增 `TrajectoryResult`，保存每条 trajectory 的核心信息：

```python
index: int
user_id: str
session_id: str
verdict: str
answer: str
response_content: str
```

评测输出中保留：

- `response`：最终送入 reformat/eval 的答案。
- `selected_index`：最终选择第几条 trajectory。
- `selected_verdict`：被选 trajectory 的 review verdict。
- `trajectories`：完整轨迹列表，用于复查。

### 答案抽取

`extract_answer()` 不直接拿完整 `response_content`，而是从 markdown block 中选择最像最终答案的部分，避免把代码输出、debug summary、q_review 文本混进投票。

评分启发包括：

- 包含 `@answer_name[...]` 的答案优先。
- 包含数字、统计关键词、final/result/summary 等内容优先。
- 排除 `[STEP GOAL]`、`Review Verdict`、debug summary 等非答案块。

### 答案签名

`answer_signature()` 用轻量 canonical form 做投票，避免纯字符串匹配过于脆弱：

- 如果答案包含 `@name[value]`，按这些格式化键值对投票。
- 否则尝试抽取 metric-name/value 对。
- 再否则抽取数字和简单分类词，如 `yes/no/linear/nonlinear/normal/significant`。
- 最后才退化为归一化文本。

### 普通 Majority Vote

`majority_vote()` 会把候选轨迹按 `answer_signature()` 分桶：

- 如果唯一最大桶存在，选择该桶里 index 最小的 trajectory。
- 如果平票，返回 `None`。
- `mv` method 在平票时 fallback 到第一条 trajectory。

### Review-Aware Vote

`review_aware_aggregate()` 的逻辑：

1. 先筛选 verdict 为 `<No_Issues>` 的轨迹。
2. 如果存在 clean 轨迹，只在 clean 轨迹中 majority vote。
3. 如果没有 clean 轨迹，则退回所有轨迹做 majority vote。
4. 如果仍然平票，fallback 到第一条 trajectory。

这样做的意图是：多数投票负责抵消随机性，`q_review` 负责降低明显有执行风险的轨迹权重。

### 评测脚本增强

`evaluation/eval_infiagent_bench.py` 现在支持：

```bash
--method baseline|q_review|mv|review_aware_vote
--trajectories 3
--vote_temperature 0.7
--trajectory_temperatures 0,0.7,1.0
--seed_results_path <existing-q-review-jsonl>
--derived_mv_result_path <write-derived-mv-jsonl>
--chat_timeout <seconds>
```

其中：

- `--seed_results_path` 可复用已有 q_review 结果作为 review-aware vote 的第 1 条轨迹，减少重复计算。
- `--derived_mv_result_path` 可在跑 review-aware vote 时，同步输出基于同一组三轨迹的普通 majority vote 结果，便于公平比较。
- `--trajectory_temperatures` 支持每条 trajectory 使用不同温度，增强多样性。

## 效果分析

实验对象为 InfiAgent-Bench dev set，共 257 道题。下面指标由官方 `eval_closed_form.py` 在 reformat 后结果上计算。

| 方法 | 题目准确率 | 子问题准确率 | 按子问题比例计分 | 正确题数 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 71.98% | 74.78% | 77.43% | 185/257 |
| q_review | 73.15% | 76.10% | 78.11% | 188/257 |
| majority vote | 73.15% | 76.10% | 77.79% | 188/257 |
| review-aware vote | 73.93% | 76.32% | 78.31% | 190/257 |

相对于 baseline：

- `q_review`：题目准确率 +1.17 个百分点，净增加 3 道正确题；有 26 道从错变对，同时 23 道从对变错。
- majority vote：题目准确率 +1.17 个百分点，净增加 3 道正确题；同样有一定随机波动。
- review-aware vote：题目准确率 +1.95 个百分点，净增加 5 道正确题，是当前几种策略中最好的。

### 结果解读

`q_review` 单独运行的提升有限，但它提供了有用的轨迹质量信号。它很保守：257 条 q_review 单轨迹中，只有 11 条被标为 `<No_Issues>`，246 条为 `<Issues_Found>`。这说明 review prompt 更偏向风险识别，而不是宽松放行。

普通 majority vote 能缓解部分随机错误，但当三条轨迹都很接近、或者错误答案形成多数时，它无法判断哪条轨迹更可靠。

review-aware vote 在同样三条轨迹上比普通 majority vote 多净胜 2 道题。它不是大幅提升，但方向符合预期：当存在 `<No_Issues>` 轨迹时，优先从 clean 轨迹里投票，可以减少部分错误轨迹对多数投票的干扰。

### 分概念表现

部分概念上的变化比较明显：

- Summary Statistics：baseline 76.67%，q_review / review-aware vote 83.33%。
- Machine Learning：baseline 31.58%，q_review 42.11%，review-aware vote 36.84%。
- Comprehensive Data Preprocessing：baseline 44.44%，q_review / review-aware vote 57.78%。
- Correlation Analysis：baseline 72.22%，review-aware vote 69.44%，说明新策略并非所有概念都稳定提升。
- Outlier Detection：baseline 65.71%，review-aware vote 71.43%。

总体看，review 机制对复杂数据处理和部分多步骤任务更有帮助，但对本来就容易通过单次计算解决的问题，额外轨迹和保守 review 可能带来波动。

### 运行成本

这组方法的成本差异很明显：

- baseline / q_review：每题 1 条 trajectory。
- majority vote / review-aware vote：默认每题 3 条 trajectory，成本接近 3 倍。
- q_review 还会在每条轨迹完成后增加一次 LLM 审查调用。

因此完整跑 257 题时，review-aware vote 的效果略好，但代价也显著更高。实际实验时建议先跑 baseline 和 q_review，再选择性跑 review-aware vote，或者用 `--seed_results_path` 复用已有 q_review 结果。

## 单题超时机制

相关文件：

- `evaluation/eval_infiagent_bench.py`
- `evaluation/review_aware_vote.py`

评测脚本新增 `--chat_timeout` 参数，用于限制一次 agent chat trajectory 的 HTTP 等待时间。`eval_infiagent_bench.py` 当前默认值是 `900` 秒；如果要严格使用单题 10 分钟上限，需要显式传：

```bash
--chat_timeout 600
```

示例：

```bash
python evaluation/eval_infiagent_bench.py \
  --method baseline \
  --note baseline-gpt4o-mini-full \
  --chat_timeout 600 \
  --num_workers 1
```

机制链路：

1. `eval_infiagent_bench.py` 将 `args.chat_timeout` 传给 `run_trajectory()`。
2. `run_trajectory()` 调用 `chat(..., timeout=chat_timeout)`。
3. `evaluation/review_aware_vote.py` 中的 `chat()` 使用 `httpx.post(..., timeout=timeout)` 调 `/chat/`。
4. 如果单条 trajectory 超过 timeout，`httpx` 抛出超时异常。
5. `run_trajectory()` 的 `finally` 块会尝试调用 `stop_session(user_id, session_id)`。
6. 当前 question 失败，不写入结果；下一次运行时，脚本会根据已写入 JSONL 的 id 自动跳过已完成题目，因此可以断点续跑。

这个 timeout 保护的是“单条 trajectory 的 HTTP chat 等待时间”。如果使用 `review_aware_vote --trajectories 3`，每道题最多会运行 3 条 trajectory，所以单题总 wall time 仍可能接近 `3 * chat_timeout`，除非某条提前完成或失败。

## Docker/Jupyter 清理机制

相关文件：

- `main.py`
- `evaluation/review_aware_vote.py`
- `evaluation/eval_infiagent_bench.py`
- `datawiseagent/agents/datawise_agent.py`
- `datawiseagent/coding/jupyter/jupyter_code_executor.py`
- `datawiseagent/coding/jupyter/docker_jupyter_server.py`
- `datawiseagent/coding/jupyter/start.sh`

### 主动停止 session

新增后端 endpoint：

```text
POST /users/{user_id}/sessions/{session_id}/stop
```

调用链路：

1. `evaluation/review_aware_vote.py::stop_session()` 发 HTTP POST 到 stop endpoint。
2. `main.py::stop_session()` 调 `agent.stop_session(session_id)`。
3. `DatawiseAgent.stop_session()` 从 `agent.sessions` 中 pop 掉 session，并调用 code executor 的 `stop()`。
4. `JupyterCodeExecutor.stop()` best-effort 停止 kernel、删除 kernel、停止 Jupyter server。
5. 如果使用 Docker Jupyter server，则进入 `DockerJupyterServer.stop()`，最终调用内部 cleanup。

`run_trajectory()` 和单文件 `review_aware_vote.py` 都在 `finally` 中调用 `stop_session()`，因此正常结束、超时、异常失败都会尝试清理资源。

### Docker 容器自动回收

在 `DatawiseAgent.create_new_session()` 中创建 Docker Jupyter server 时设置：

```python
DockerJupyterServer(
    auto_remove=True,
    stop_container=True,
    ...
)
```

含义：

- `stop_container=True`：注册 cleanup，在 session stop 或进程退出时停止容器。
- `auto_remove=True`：容器停止后由 Docker 自动删除，避免长时间跑 benchmark 后残留大量 `jupyterkernelgateway-*` 容器。

`DockerJupyterServer` 内部还会：

- 为每个 session 生成唯一容器名 `jupyterkernelgateway-{uuid}`。
- 将 session workspace 挂载到容器内 `/mnt`。
- 等待 kernel gateway 启动；如果启动超时或异常，会 stop/remove 容器。

### Jupyter gateway 启动脚本

`datawiseagent/coding/jupyter/start.sh` 是容器内启动 kernel gateway 的脚本。它会先检查容器内 8888 端口是否被占用：

```bash
if lsof -i:8888; then
    lsof -ti:8888 | xargs kill -9
fi
```

然后启动：

```bash
python -m jupyter kernelgateway ...
```

这个脚本解决的是容器内部 gateway 端口冲突问题；主机侧 Docker 容器清理则由 stop endpoint、`JupyterCodeExecutor.stop()` 和 `DockerJupyterServer.cleanup()` 负责。

## 常用运行命令

启动后端：

```bash
cd /mnt/data/fulin/510_bighw/DatawiseAgent
python main.py
```

跑 baseline，严格单条 trajectory 10 分钟超时：

```bash
python evaluation/eval_infiagent_bench.py \
  --method baseline \
  --note baseline-gpt4o-mini-full \
  --chat_timeout 600 \
  --num_workers 1
```

跑 q_review：

```bash
python evaluation/eval_infiagent_bench.py \
  --method q_review \
  --note q_review-gpt4o-mini-full \
  --chat_timeout 600 \
  --num_workers 1
```

跑 review-aware vote，并同步导出同轨迹的普通 majority vote：

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

重格式化答案：

```bash
cd evaluation
python InfiAgentBench/scripts/reformat.py \
  --model gpt-4o-mini \
  --responses_file_path experimental_results/InfiAgent-Bench/results_review_aware_vote_gpt4o-mini-full.jsonl \
  --output_file_path experimental_results/InfiAgent-Bench/reformat/results_reformat_review_aware_vote-gpt4o-mini-full.jsonl
```

`reformat.py` 会把格式化答案写入 `reformat_response` 字段，而官方 `eval_closed_form.py` 读取的是 `response` 字段。因此正式计算指标前，需要生成一个 eval 版本，把原始回答保存在 `raw_response`，并将 `response` 替换为 `reformat_response`。当前仓库中的 `results_eval_*` 文件就是这种格式。

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

计算 closed-form accuracy：

```bash
cd evaluation
python InfiAgentBench/scripts/eval_closed_form.py \
  --questions_file_path InfiAgentBench/data/da-dev-questions.jsonl \
  --labels_file_path InfiAgentBench/data/da-dev-labels.jsonl \
  --responses_file_path experimental_results/InfiAgent-Bench/reformat/results_eval_review_aware_vote-gpt4o-mini-full.jsonl
```

## 当前结论

`q_review` 和 majority vote 都能带来小幅提升，但不是免费收益。`q_review` 的主要价值是提供轨迹可信度信号；单独作为最终答案选择器时效果有限。普通 majority vote 能缓解随机性，但无法判断错误多数。review-aware vote 把两者结合后，在当前完整 257 题结果上表现最好，题目准确率从 baseline 的 71.98% 提升到 73.93%。

工程层面，单题 timeout 和主动 stop session 是跑完整 benchmark 的必要保护。否则长时间运行时，失败题会卡住 HTTP 请求，Docker/Jupyter 容器也可能持续占用资源，导致后续题目越来越慢甚至无法启动新 session。
