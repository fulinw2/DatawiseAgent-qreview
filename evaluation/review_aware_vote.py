"""
Run self-consistency trajectories and aggregate them with q_review verdicts.

Prerequisite:
    Start the backend server first, e.g. `python main.py`.

Example:
    python evaluation/review_aware_vote.py \
        --query "Question: ..." \
        --file evaluation/InfiAgentBench/data/da-dev-tables/baro_2015.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datawiseagent.agents.datawise_agent import agent_config, llm_config
from datawiseagent.prompts.datawise import ISSUES_FOUND_TAG, NO_ISSUES_TAG


BASE_URL = "http://0.0.0.0:8000"


@dataclass
class TrajectoryResult:
    index: int
    user_id: str
    session_id: str
    verdict: str
    answer: str
    response_content: str


def _post_json(path: str, payload: dict, timeout: float | None = None) -> dict:
    response = httpx.post(f"{BASE_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def create_user(username: str) -> str:
    return _post_json("/create_user/", {"username": username})["user_id"]


def create_session(
    user_id: str,
    session_name: str,
    reset_llm_config: dict,
    tool_mode: str = "default",
) -> str:
    payload = {
        "tool_mode": tool_mode,
        "reset_llm_config": reset_llm_config,
    }
    data = _post_json(f"/users/{user_id}/sessions/{session_name}", payload)
    return data["session_id"]


def upload_files(user_id: str, session_id: str, files: Iterable[Path]) -> None:
    for path in files:
        with path.open("rb") as f:
            response = httpx.post(
                f"{BASE_URL}/upload/",
                data={"user_id": user_id, "session_id": session_id},
                files={"files": (path.name, f)},
                timeout=None,
            )
        response.raise_for_status()


def stop_session(user_id: str, session_id: str) -> None:
    response = httpx.post(
        f"{BASE_URL}/users/{user_id}/sessions/{session_id}/stop", timeout=60
    )
    if response.status_code != 404:
        response.raise_for_status()


def chat(
    user_id: str,
    session_id: str,
    query: str,
    run_agent_config: dict,
    work_mode: str = "jupyter",
    timeout: float | None = 3600,
) -> str:
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "query": query,
        "agent_config": run_agent_config,
        "work_mode": work_mode,
    }
    data = _post_json("/chat/", payload, timeout=timeout)
    return data["response_content"]


def extract_verdict(response_content: str) -> str:
    explicit_matches = re.findall(
        r"Review\s+Verdict:\s*(<No_Issues>|<Issues_Found>)",
        response_content,
        flags=re.I,
    )
    if explicit_matches:
        normalized = {match.lower(): match for match in explicit_matches}
        if len(normalized) == 1:
            only_match = next(iter(normalized.values()))
            return (
                NO_ISSUES_TAG
                if only_match.lower() == NO_ISSUES_TAG.lower()
                else ISSUES_FOUND_TAG
            )
        return ISSUES_FOUND_TAG

    leading_match = re.search(
        r"^\s*(<No_Issues>|<Issues_Found>)", response_content, flags=re.I
    )
    if leading_match:
        verdict = leading_match.group(1)
        return (
            NO_ISSUES_TAG
            if verdict.lower() == NO_ISSUES_TAG.lower()
            else ISSUES_FOUND_TAG
        )

    mentions_no_issues = re.search(re.escape(NO_ISSUES_TAG), response_content, re.I)
    mentions_issues = re.search(re.escape(ISSUES_FOUND_TAG), response_content, re.I)
    if mentions_no_issues and not mentions_issues:
        return NO_ISSUES_TAG
    if mentions_issues and not mentions_no_issues:
        return ISSUES_FOUND_TAG
    return ISSUES_FOUND_TAG


def _strip_review(response_content: str) -> str:
    return re.split(r"```markdown\s*Review Verdict:", response_content, maxsplit=1)[0]


def _is_answer_candidate(block: str) -> bool:
    lowered = block.lower()
    if not block.strip():
        return False
    if "[step goal]" in lowered or "review verdict:" in lowered:
        return False
    if "debugging summary" in lowered:
        return False
    return True


def _score_answer_candidate(block: str) -> int:
    lowered = block.lower()
    score = 0
    if re.search(r"@\w+\[.*?\]", block):
        score += 8
    if re.search(r"\d", block):
        score += 4
    if any(
        token in lowered
        for token in (
            "answer",
            "result",
            "summary",
            "conclusion",
            "final",
            "mean",
            "median",
            "standard deviation",
            "std",
            "coefficient",
            "p-value",
            "accuracy",
            "outlier",
            "linear",
            "normal",
        )
    ):
        score += 3
    if "further questions" in lowered or "please let me know" in lowered:
        score -= 3
    if len(block.split()) < 8 and not re.search(r"@\w+\[.*?\]", block):
        score -= 1
    return score


def extract_answer(response_content: str) -> str:
    """
    Keep aggregation focused on the final substantive answer, not on execution
    traces, q_review, or a trailing courtesy sentence.
    """
    answer_region = _strip_review(response_content)
    markdown_blocks = re.findall(r"```markdown\s*(.*?)```", answer_region, flags=re.S)
    candidates = [
        block.strip() for block in markdown_blocks if _is_answer_candidate(block)
    ]
    if candidates:
        indexed = list(enumerate(candidates))
        _, best = max(
            indexed,
            key=lambda item: (_score_answer_candidate(item[1]), item[0]),
        )
        return best
    return answer_region.strip()


def normalize_answer(answer: str) -> str:
    answer = re.sub(r"```.*?```", " ", answer, flags=re.S)
    answer = re.sub(r"\s+", " ", answer).strip().lower()
    return answer


def answer_signature(answer: str) -> str:
    """
    Use a light canonical form for voting. Final answers are often equivalent
    but phrased differently, so exact prose matching is too brittle.
    """
    formatted_pairs = re.findall(r"@(\w+)\[(.*?)\]", answer)
    if formatted_pairs:
        return json.dumps(
            [(name.lower(), value.strip().lower()) for name, value in formatted_pairs],
            ensure_ascii=False,
        )

    number_pattern = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    metric_pairs = re.findall(
        rf"(?i)\b("
        rf"mean|median|standard deviation|std(?:\s*dev(?:iation)?)?|"
        rf"correlation coefficient|coefficient|p[- ]?value|accuracy|"
        rf"total(?:\s+number)?(?:\s+of)?(?:\s+[a-z_ -]+)?|"
        rf"count|outliers?"
        rf")\b[^:\n=]*[:=]\s*\$?\*?({number_pattern})",
        answer,
    )
    if metric_pairs:
        return json.dumps(
            [
                (name.lower().strip(), value.replace(",", ""))
                for name, value in metric_pairs
            ],
            ensure_ascii=False,
        )

    evidence_lines = []
    for line in answer.splitlines():
        lowered = line.lower()
        if any(
            token in lowered
            for token in (
                "mean",
                "median",
                "standard deviation",
                "std",
                "coefficient",
                "p-value",
                "accuracy",
                "outlier",
                "total",
                "count",
                "number",
                "range",
                "normal",
                "linear",
                "nonlinear",
                "significant",
                "relationship",
                "result",
            )
        ):
            evidence_lines.append(line)
    evidence_text = "\n".join(evidence_lines) if evidence_lines else answer
    evidence_text = re.sub(r"(?m)^\s*\d+\.\s+", "", evidence_text)
    numbers = [
        value.replace(",", "") for value in re.findall(number_pattern, evidence_text)
    ]
    categories = re.findall(
        r"\b(?:yes|no|linear|nonlinear|normal|not normal|significant|"
        r"no significant correlation)\b",
        evidence_text.lower(),
    )
    if numbers or categories:
        return json.dumps({"numbers": numbers, "categories": categories})
    return normalize_answer(answer)


def majority_vote(candidates: list[TrajectoryResult]) -> TrajectoryResult | None:
    if not candidates:
        return None
    buckets: dict[str, list[TrajectoryResult]] = {}
    for candidate in candidates:
        buckets.setdefault(answer_signature(candidate.answer), []).append(candidate)
    best_size = max(len(items) for items in buckets.values())
    winners = [items for items in buckets.values() if len(items) == best_size]
    if len(winners) == 1:
        return sorted(winners[0], key=lambda item: item.index)[0]
    return None


def review_aware_aggregate(results: list[TrajectoryResult]) -> TrajectoryResult:
    clean = [item for item in results if item.verdict == NO_ISSUES_TAG]
    voted = majority_vote(clean if clean else results)
    if voted is not None:
        return voted
    return results[0]


def run(args: argparse.Namespace) -> dict:
    files = [Path(path) for path in args.file]
    user_id = create_user(args.username)
    run_agent_config = deepcopy(agent_config)
    run_agent_config.setdefault("review", {})["enabled"] = not args.disable_review

    results: list[TrajectoryResult] = []
    for index in range(1, args.trajectories + 1):
        run_llm_config = deepcopy(llm_config)
        run_llm_config["temperature"] = args.temperature
        session_id = create_session(
            user_id,
            f"{args.session_prefix}-{index}",
            run_llm_config,
            tool_mode=args.tool_mode,
        )
        try:
            upload_files(user_id, session_id, files)
            response_content = chat(
                user_id,
                session_id,
                args.query,
                run_agent_config,
                work_mode=args.work_mode,
                timeout=args.chat_timeout,
            )
        finally:
            try:
                stop_session(user_id, session_id)
            except Exception:
                pass
        results.append(
            TrajectoryResult(
                index=index,
                user_id=user_id,
                session_id=session_id,
                verdict=extract_verdict(response_content),
                answer=extract_answer(response_content),
                response_content=response_content,
            )
        )

    selected = review_aware_aggregate(results)
    return {
        "selected_index": selected.index,
        "selected_verdict": selected.verdict,
        "selected_answer": selected.answer,
        "trajectories": [asdict(item) for item in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--username", default="review-aware-vote")
    parser.add_argument("--session-prefix", default="trajectory")
    parser.add_argument("--tool-mode", default="default")
    parser.add_argument("--work-mode", default="jupyter")
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--chat-timeout", type=float, default=3600)
    parser.add_argument("--disable-review", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = run(args)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if output_path := args.output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
