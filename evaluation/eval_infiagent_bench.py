"""
Evaluate DatawiseAgent on InfiAgentBench.

Start the FastAPI backend before running this script:
    python ../main.py

Run from either the repository root or the evaluation directory:
    python evaluation/eval_infiagent_bench.py --method baseline --note smoke --limit 1
    python ./eval_infiagent_bench.py --method review_aware_vote --note mv-r

The field `response` is the final answer that should be sent to the official
InfiAgentBench reformat/evaluation scripts. Full trajectories are retained in
`trajectories` for auditability.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import uuid
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from review_aware_vote import (
        TrajectoryResult,
        chat,
        create_session,
        create_user,
        extract_answer,
        extract_verdict,
        majority_vote,
        review_aware_aggregate,
        stop_session,
        upload_files,
    )
except ModuleNotFoundError:
    from evaluation.review_aware_vote import (
        TrajectoryResult,
        chat,
        create_session,
        create_user,
        extract_answer,
        extract_verdict,
        majority_vote,
        review_aware_aggregate,
        stop_session,
        upload_files,
    )
from datawiseagent.agents.datawise_agent import agent_config, llm_config


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_PATH = EVALUATION_DIR / "InfiAgentBench/data/da-dev-questions.jsonl"
DEFAULT_TABLE_DIR = EVALUATION_DIR / "InfiAgentBench/data/da-dev-tables"
DEFAULT_RESULTS_DIR = EVALUATION_DIR / "experimental_results/InfiAgent-Bench"
NO_REVIEW_TAG = "<No_Review>"

Method = Literal["baseline", "q_review", "mv", "review_aware_vote"]


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def existing_result_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()

    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in item:
                ids.add(item["id"])
    return ids


def load_seed_trajectories(path: Path | None) -> dict[int, TrajectoryResult]:
    if path is None or not path.exists():
        return {}

    seeds: dict[int, TrajectoryResult] = {}
    for item in read_jsonl(path):
        question_id = item.get("id")
        if question_id is None or question_id in seeds:
            continue

        trajectory = (item.get("trajectories") or [{}])[0]
        response_content = trajectory.get("response_content") or item.get(
            "response", ""
        )
        answer = trajectory.get("answer") or item.get("response", "")
        verdict = trajectory.get("verdict") or extract_verdict(response_content)
        seeds[int(question_id)] = TrajectoryResult(
            index=1,
            user_id=str(trajectory.get("user_id") or item.get("user_id") or ""),
            session_id=str(
                trajectory.get("session_id") or item.get("session_id") or ""
            ),
            verdict=verdict,
            answer=answer,
            response_content=response_content,
        )
    return seeds


def parse_temperature_schedule(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def temperature_for_index(args: argparse.Namespace, index: int) -> float | None:
    if args.trajectory_temperatures:
        schedule = args.trajectory_temperatures
        if index <= len(schedule):
            return schedule[index - 1]
        return schedule[-1]
    if args.method in ("mv", "review_aware_vote"):
        return args.vote_temperature
    return args.temperature


def build_prompt(question: dict) -> str:
    return f"Question: {question['question']}\n{question['constraints']}\n"


def build_agent_config(review_enabled: bool) -> dict:
    run_agent_config = deepcopy(agent_config)
    run_agent_config.setdefault("review", {})["enabled"] = review_enabled
    return run_agent_config


def build_llm_config(temperature: float | None) -> dict:
    run_llm_config = deepcopy(llm_config)
    if temperature is not None:
        run_llm_config["temperature"] = temperature
    return run_llm_config


def trajectory_count(method: Method, trajectories: int) -> int:
    if method in ("mv", "review_aware_vote"):
        return trajectories
    return 1


def review_enabled_for_method(method: Method) -> bool:
    return method in ("q_review", "review_aware_vote")


def aggregate_results(
    method: Method, results: list[TrajectoryResult]
) -> TrajectoryResult:
    if method == "review_aware_vote":
        return review_aware_aggregate(results)
    if method == "mv":
        return majority_vote(results) or results[0]
    return results[0]


def build_result_record(
    *,
    question: dict,
    prompt: str,
    file_path: Path,
    method: Method,
    aggregation: str,
    selected: TrajectoryResult,
    trajectories: list[TrajectoryResult],
    user_id: str,
    selected_verdict: str | None = None,
) -> dict:
    return {
        "id": question["id"],
        "input_text": prompt,
        "concepts": question["concepts"],
        "file_path": str(file_path),
        "response": selected.answer,
        "format": question["format"],
        "method": method,
        "aggregation": aggregation,
        "selected_index": selected.index,
        "selected_verdict": selected_verdict or selected.verdict,
        "user_id": str(user_id),
        "session_id": selected.session_id,
        "trajectories": [asdict(item) for item in trajectories],
    }


def run_trajectory(
    *,
    user_id: str,
    question: dict,
    table_path: Path,
    method: Method,
    trajectory_index: int,
    temperature: float | None,
    work_mode: str,
    chat_timeout: float | None,
) -> TrajectoryResult:
    file_name = question["file_name"]
    file_path = table_path / file_name
    prompt = build_prompt(question)
    input_prompt = prompt + "\n" + f"{file_name} has been uploaded."

    run_agent_config = build_agent_config(review_enabled_for_method(method))
    run_llm_config = build_llm_config(temperature)
    session_name = f"{method}-q{question['id']}-t{trajectory_index}"
    session_id = create_session(
        user_id=user_id,
        session_name=session_name,
        reset_llm_config=run_llm_config,
        tool_mode="default",
    )
    try:
        upload_files(user_id, session_id, [file_path])
        response_content = chat(
            user_id=user_id,
            session_id=session_id,
            query=input_prompt,
            run_agent_config=run_agent_config,
            work_mode=work_mode,
            timeout=chat_timeout,
        )
    finally:
        try:
            stop_session(user_id, session_id)
        except Exception:
            pass

    review_enabled = review_enabled_for_method(method)
    return TrajectoryResult(
        index=trajectory_index,
        user_id=user_id,
        session_id=session_id,
        verdict=extract_verdict(response_content) if review_enabled else NO_REVIEW_TAG,
        answer=extract_answer(response_content),
        response_content=response_content,
    )


def process_question(question: dict, args: argparse.Namespace) -> tuple[dict, dict | None]:
    method: Method = args.method
    user_id = create_user(
        username=f"InfiAgentBench-{method}-q{question['id']}-{uuid.uuid4()}"
    )
    prompt = build_prompt(question)
    table_path = args.table_path
    file_path = table_path / question["file_name"]
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing table file for question {question['id']}: {file_path}"
        )

    n_trajectories = trajectory_count(method, args.trajectories)

    trajectories: list[TrajectoryResult] = []
    seed = args.seed_trajectories.get(question["id"])
    if seed is not None and method == "review_aware_vote":
        trajectories.append(seed)

    for index in range(len(trajectories) + 1, n_trajectories + 1):
        trajectories.append(
            run_trajectory(
                user_id=user_id,
                question=question,
                table_path=table_path,
                method=method,
                trajectory_index=index,
                temperature=temperature_for_index(args, index),
                work_mode=args.work_mode,
                chat_timeout=args.chat_timeout,
            )
        )

    selected = aggregate_results(method, trajectories)
    result = build_result_record(
        question=question,
        prompt=prompt,
        file_path=file_path,
        method=method,
        aggregation=(
            "review_aware_vote"
            if method == "review_aware_vote"
            else ("majority_vote" if method == "mv" else "-")
        ),
        selected=selected,
        trajectories=trajectories,
        user_id=user_id,
    )

    derived_mv = None
    if method == "review_aware_vote" and args.derived_mv_result_path is not None:
        mv_selected = majority_vote(trajectories) or trajectories[0]
        derived_mv = build_result_record(
            question=question,
            prompt=prompt,
            file_path=file_path,
            method="mv",
            aggregation="majority_vote",
            selected=mv_selected,
            trajectories=trajectories,
            user_id=user_id,
            selected_verdict=NO_REVIEW_TAG,
        )

    return result, derived_mv


def select_questions(
    questions: list[dict], args: argparse.Namespace, completed_ids: set[int]
) -> list[dict]:
    selected = []
    requested_ids = set(args.ids or [])
    for question in questions:
        if requested_ids and question["id"] not in requested_ids:
            continue
        if args.level and question.get("level") != args.level:
            continue
        if question["id"] in completed_ids and not args.rerun:
            continue
        selected.append(question)
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected


def write_result(path: Path, result: dict) -> None:
    with path.open("a+", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()


def run(args: argparse.Namespace) -> None:
    args.questions_path = Path(args.questions_path)
    args.table_path = Path(args.table_path)
    args.seed_results_path = (
        Path(args.seed_results_path) if args.seed_results_path is not None else None
    )
    args.seed_trajectories = load_seed_trajectories(args.seed_results_path)
    args.trajectory_temperatures = parse_temperature_schedule(
        args.trajectory_temperatures
    )
    args.derived_mv_result_path = (
        Path(args.derived_mv_result_path)
        if args.derived_mv_result_path is not None
        else None
    )
    if args.result_path is None:
        note = args.note or f"{args.method}-temperature-{llm_config.get('temperature')}"
        args.result_path = DEFAULT_RESULTS_DIR / f"results_{note}.jsonl"
    else:
        args.result_path = Path(args.result_path)

    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    if args.rerun and args.result_path.exists():
        args.result_path.unlink()
    args.result_path.touch(exist_ok=True)
    if args.derived_mv_result_path is not None:
        args.derived_mv_result_path.parent.mkdir(parents=True, exist_ok=True)
        if args.rerun and args.derived_mv_result_path.exists():
            args.derived_mv_result_path.unlink()
        args.derived_mv_result_path.touch(exist_ok=True)
    args.derived_mv_completed_ids = (
        existing_result_ids(args.derived_mv_result_path)
        if args.derived_mv_result_path is not None
        else set()
    )

    questions = read_jsonl(args.questions_path)
    completed_ids = existing_result_ids(args.result_path)
    pending_questions = select_questions(questions, args, completed_ids)

    print(f"method={args.method}")
    print(f"questions_path={args.questions_path}")
    print(f"table_path={args.table_path}")
    print(f"result_path={args.result_path}")
    if args.seed_results_path is not None:
        print(f"seed_results_path={args.seed_results_path}")
        print(f"seed_trajectories={len(args.seed_trajectories)}")
    if args.derived_mv_result_path is not None:
        print(f"derived_mv_result_path={args.derived_mv_result_path}")
    print(f"pending={len(pending_questions)} / total={len(questions)}")

    if args.num_workers == 1:
        for question in pending_questions:
            try:
                result, derived_mv = process_question(question, args)
                write_result(args.result_path, result)
                if derived_mv is not None and (
                    not args.derived_mv_completed_ids
                    or derived_mv["id"] not in args.derived_mv_completed_ids
                    or args.rerun
                ):
                    write_result(args.derived_mv_result_path, derived_mv)
                print(
                    f"written id={result['id']} selected={result['selected_index']} "
                    f"verdict={result['selected_verdict']}"
                )
            except Exception:
                print(f"failed id={question.get('id')}")
                traceback.print_exc()
        return

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_question = {
            executor.submit(process_question, question, args): question
            for question in pending_questions
        }
        for future in as_completed(future_to_question):
            question = future_to_question[future]
            try:
                result, derived_mv = future.result()
                write_result(args.result_path, result)
                if derived_mv is not None and (
                    not args.derived_mv_completed_ids
                    or derived_mv["id"] not in args.derived_mv_completed_ids
                    or args.rerun
                ):
                    write_result(args.derived_mv_result_path, derived_mv)
                print(
                    f"written id={result['id']} selected={result['selected_index']} "
                    f"verdict={result['selected_verdict']}"
                )
            except Exception:
                print(f"failed id={question.get('id')}")
                traceback.print_exc()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", type=str, help="Readable identifier for this run.")
    parser.add_argument(
        "--method",
        choices=["baseline", "q_review", "mv", "review_aware_vote"],
        default="baseline",
    )
    parser.add_argument("--questions_path", default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--table_path", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--result_path", type=Path)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override single-trajectory temperature; default uses config.",
    )
    parser.add_argument(
        "--vote_temperature",
        type=float,
        default=0.7,
        help="Temperature for mv/review_aware_vote trajectories.",
    )
    parser.add_argument(
        "--trajectory_temperatures",
        type=str,
        help=(
            "Comma-separated per-trajectory temperatures, e.g. 0.2,0.7,1.0. "
            "Overrides --temperature/--vote_temperature for listed indices."
        ),
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        default=3,
        help="Number of trajectories for mv/review_aware_vote.",
    )
    parser.add_argument(
        "--seed_results_path",
        type=Path,
        help=(
            "Existing q_review/review_aware result JSONL to reuse as trajectory 1 "
            "for review_aware_vote."
        ),
    )
    parser.add_argument(
        "--derived_mv_result_path",
        type=Path,
        help=(
            "When running review_aware_vote, also write ordinary majority-vote "
            "records derived from the same trajectories."
        ),
    )
    parser.add_argument(
        "--chat_timeout",
        type=float,
        default=900,
        help="HTTP timeout in seconds for one agent chat trajectory.",
    )
    parser.add_argument(
        "--work_mode", default="jupyter", choices=["jupyter", "jupyter+script"]
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", type=int, action="append")
    parser.add_argument("--level", choices=["easy", "medium", "hard"])
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Overwrite the result file and recompute selected questions.",
    )
    args = parser.parse_args()
    if args.trajectories < 1:
        parser.error("--trajectories must be >= 1")
    if args.num_workers < 1:
        parser.error("--num_workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


if __name__ == "__main__":
    run(parse_args())
