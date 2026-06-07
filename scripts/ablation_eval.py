import argparse
import json
import sys
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [script_path.parent, *script_path.parents]

    for candidate in candidates:
        if (candidate / "core").is_dir() and (candidate / "browsecomp_plus_hard50.jsonl").exists():
            return candidate

    for candidate in candidates:
        if (candidate / "core").is_dir():
            return candidate

    return Path.cwd()


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agent.dataset_utils import load_jsonl
from core.agent.eval import run_evaluation
from core.agent.tools import (
    build_searcher,
    get_basic_tool_specs_and_registry,
    get_document_window_tool_specs_and_registry,
    get_search_tool_specs_and_registry,
)
from core.agent.vllm_client import VLLMClient
from core.agent.agent import (
    add_confirmed_facts,
    compact_result,
    extract_json_object,
    format_final_answer as format_basic_final_answer,
    normalize_confidence,
    normalize_query,
    result_key,
    summarize_older_rounds,
)
import open_track.agent.research_agent as open_track_agent


OPEN_TRACK_NO_WINDOW_LOOP_PROMPT = """You are loop_agent for a simple deep research agent.

You will receive the current work_message: original question, constraints, evident snippets,
candidate answers, tool history, and previous visible analyses.

Your response content is your current visible analysis. Keep it concise and useful.
Decide what to investigate next from the current state.

Available tool:
- search(query): discover candidate documents, bridge entities, names, titles, dates, source pages,
  or documents containing distinctive clues.

Tool-use rules:
- You must call 2-4 search tools when more investigation is useful.
- Never call get_document_window; it is disabled for this ablation run.
- Never search the full original question.
- Do not write natural-language search questions.
- Use compact high-information search terms.
- Prefer discovered bridge entities, titles, names, single exact years when distinctive,
  distinctive phrases, and relation anchors.
- Never include numeric ranges, decade expressions, or page ranges in search queries.
- Search queries should combine rare terms from one coherent clue cluster.
- Avoid repeating equivalent tool calls from tool_history.
- If you use <think>...</think>, you must write a short visible analysis after </think> before any tool call.
- Your post-think visible content must not be empty. Use:
  <think>private reasoning</think>
  Analysis: concise reason for the next search call or final answer.

Final-answer rules:
- If a candidate answer has direct evidence, all critical constraints are supported, most strong constraints
  are supported, and there is no contradiction, stop calling tools and answer in English.
- Final answer must include brief evidence with docids, an `Exact Answer:` line, and a `Confidence:` line.
"""


BASIC_WITH_WINDOW_LOOP_PROMPT = """You are a single-agent BrowseComp-Plus research agent for an ablation experiment.

Task:
- Answer the original question using only provided tool results.
- The controller can execute two tools for you:
  1. search(query): BM25 search returning docid, score, url, and a leading snippet.
  2. get_document_window(docid, keyword): return windows around up to the first three occurrences
     of one single-word keyword inside a known document.
- After each tool round, decide whether the evidence is sufficient.
- If evidence is insufficient, choose the next 1-3 tool actions.

Evidence rules:
- Do not answer from memory.
- Treat a fact as confirmed only when it is directly supported by a provided snippet/window.
- evidence_sufficient must be true only when the answer is directly supported and key constraints
  needed to identify the answer are supported.
- If snippets are off-topic, generic, contradictory, or do not contain the target answer, set
  evidence_sufficient to false.

Tool rules:
- Use search to discover unknown entities, titles, people, organizations, source pages, or bridge documents.
- Use get_document_window when a promising docid is already known and the missing evidence is likely
  inside that document: table of contents, chapter title, acknowledgements, dedication, names, dates,
  prices, page-like clues, object descriptions, or rare phrases.
- get_document_window keyword must be exactly one word with no spaces, such as "contents",
  "chapter", "acknowledgements", "dedication", "married", "spear", "barrel", or a surname.
- Never search the full original question as a rewritten query.
- Do not write natural-language search questions.
- Search queries should be compact high-information terms, usually 4-10 terms.
- Avoid broad generic queries and avoid repeating previous actions.

Return strict JSON only. No markdown.
Schema:
{
  "analysis": "brief reasoning about the latest evidence",
  "evidence_sufficient": false,
  "final_answer": "",
  "confidence": "low",
  "used_docids": [],
  "key_facts": [
    "directly supported fact with docid"
  ],
  "actions": [
    {"tool": "search", "query": "compact query"},
    {"tool": "get_document_window", "docid": "123", "keyword": "contents"}
  ]
}

When evidence_sufficient is true:
- final_answer must be non-empty.
- used_docids must list docids that directly support the answer.
- actions must be empty.
- confidence should be medium or high.

When evidence_sufficient is false:
- final_answer should be empty unless there is a weak candidate that still needs verification.
- actions should contain 1-3 useful next actions unless no useful action exists.
"""


BASIC_WITH_WINDOW_FINAL_PROMPT = """You are a single-agent deep research agent at the final answer stage.

The search/window loop has stopped before reaching a fully verified answer. You must now give the
most likely answer based only on the provided tool results, confirmed facts, candidate answers, and
round summaries.

Rules:
- Do not use outside knowledge.
- Prefer an answer string that appears directly in snippets or document windows.
- If no answer string is directly supported, choose the most plausible candidate from the evidence
  and make the uncertainty clear in the explanation.
- Do not output NOT FOUND unless the tool results contain no plausible candidate at all.
- Return strict JSON only. No markdown.

Schema:
{
  "analysis": "brief explanation of why this is the most likely answer and what remains uncertain",
  "final_answer": "most likely answer",
  "confidence": "low|medium|high",
  "used_docids": ["docid"]
}
"""


VARIANTS = ("open_track_no_window", "basic_with_window")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ablation experiments on BrowseComp-Plus hard50: "
            "OpenTrack without document-window and baseline/basic with document-window."
        )
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="Ablation variants to run.",
    )
    parser.add_argument("--dataset", default="browsecomp_plus_hard50.jsonl", help="BrowseComp-Plus hard50 JSONL path.")
    parser.add_argument("--index-path", default="indexes/browsecomp_plus_bm25.sqlite", help="BM25 SQLite index path.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI-compatible base URL.")
    parser.add_argument("--model", default="qwen_auto", help="Model name for agents and default judge.")
    parser.add_argument("--eval-model", default=None, help="Judge model name. Defaults to --model.")
    parser.add_argument("--api-key", default="dummy", help="API key for the vLLM endpoint.")
    parser.add_argument("--top-k", type=int, default=6, help="Search results per search call.")
    parser.add_argument("--snippet-max-chars", type=int, default=1600, help="Maximum characters per search snippet.")
    parser.add_argument("--window-chars", type=int, default=1200, help="Characters returned by get_document_window.")
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum agent loop rounds.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="max_tokens for agent model calls.")
    parser.add_argument("--eval-max-tokens", type=int, default=4096, help="max_tokens for judge calls.")
    parser.add_argument("--eval-max-workers", type=int, default=8, help="Parallel judge workers.")
    parser.add_argument("--recent-rounds", type=int, default=3, help="Basic-with-window: recent rounds kept in full.")
    parser.add_argument(
        "--context-snippet-chars",
        type=int,
        default=1200,
        help="Basic-with-window: maximum snippet/window characters shown per result.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of dataset rows to run. Defaults to all hard50.")
    parser.add_argument("--output-dir", default="runs", help="Directory for ablation files.")
    parser.add_argument("--timestamp", default=None, help="Optional fixed timestamp for output filenames.")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Do not embed full submission messages in the combined ablation JSON file.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only run predictions and write submissions; skip judge evaluation.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


@contextmanager
def patched_open_track_loop_prompt(prompt: str) -> Iterable[None]:
    old_prompt = open_track_agent.LOOP_PROMPT
    open_track_agent.LOOP_PROMPT = prompt
    try:
        yield
    finally:
        open_track_agent.LOOP_PROMPT = old_prompt


def make_tool_call(call_id: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def action_signature(action: Dict[str, str]) -> str:
    if action.get("tool") == "search":
        return f"search:{normalize_query(action.get('query', '')).lower()}"
    return (
        f"get_document_window:{str(action.get('docid', '')).strip()}:"
        f"{str(action.get('keyword', '')).strip().lower()}"
    )


def normalize_actions(raw_actions: Any, max_actions: int = 3) -> List[Dict[str, str]]:
    actions: List[Dict[str, str]] = []
    if not isinstance(raw_actions, list):
        return actions

    seen = set()
    for raw_action in raw_actions:
        if not isinstance(raw_action, dict):
            continue
        tool = str(raw_action.get("tool") or raw_action.get("name") or raw_action.get("tool_name") or "").strip()
        if tool == "search":
            query = normalize_query(raw_action.get("query", ""))
            if not query:
                continue
            action = {"tool": "search", "query": query}
        elif tool == "get_document_window":
            docid = str(raw_action.get("docid", "")).strip()
            keyword = str(raw_action.get("keyword", "")).strip()
            if not docid or not keyword or len(keyword.split()) != 1:
                continue
            action = {"tool": "get_document_window", "docid": docid, "keyword": keyword}
        else:
            continue

        signature = action_signature(action)
        if signature in seen:
            continue
        actions.append(action)
        seen.add(signature)
        if len(actions) >= max_actions:
            break
    return actions


def normalize_window_decision(raw_decision: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    decision = raw_decision if isinstance(raw_decision, dict) else {}
    key_facts = decision.get("key_facts", [])
    if not isinstance(key_facts, list):
        key_facts = []
    used_docids = decision.get("used_docids", [])
    if not isinstance(used_docids, list):
        used_docids = []

    actions = normalize_actions(decision.get("actions", []))
    next_query = normalize_query(decision.get("next_query", ""))
    if not actions and next_query:
        actions = [{"tool": "search", "query": next_query}]

    return {
        "analysis": str(decision.get("analysis", "")).strip(),
        "evidence_sufficient": bool(decision.get("evidence_sufficient", False)),
        "final_answer": str(decision.get("final_answer", "")).strip(),
        "confidence": normalize_confidence(decision.get("confidence")),
        "used_docids": [str(docid) for docid in used_docids if str(docid).strip()],
        "key_facts": [str(fact).strip() for fact in key_facts if str(fact).strip()],
        "actions": actions,
    }


def compact_tool_result(result: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    compact = compact_result(result, max_chars)
    for key in ("found", "keyword", "error"):
        if key in result:
            compact[key] = result[key]
    return compact


def summarize_window_rounds(rounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for item in rounds:
        decision = item.get("decision", {})
        summary.append(
            {
                "round": item.get("round"),
                "actions": item.get("actions", []),
                "docids": [str(result.get("docid", "")) for result in item.get("results", [])],
                "new_docids": item.get("new_docids", []),
                "analysis": str(decision.get("analysis", ""))[:500],
                "key_facts": decision.get("key_facts", [])[:5],
            }
        )
    return summary


def build_window_decision_messages(
    question: str,
    tool_rounds: List[Dict[str, Any]],
    confirmed_facts: List[str],
    previous_actions: List[Dict[str, str]],
    round_id: int,
    max_rounds: int,
    recent_rounds: int,
    context_snippet_chars: int,
) -> List[Dict[str, str]]:
    older = tool_rounds[:-recent_rounds] if recent_rounds > 0 else tool_rounds
    recent = tool_rounds[-recent_rounds:] if recent_rounds > 0 else []
    recent_payload = []
    for item in recent:
        recent_payload.append(
            {
                "round": item.get("round"),
                "actions": item.get("actions", []),
                "new_docids": item.get("new_docids", []),
                "documents": [
                    compact_tool_result(result, context_snippet_chars)
                    for result in item.get("results", [])
                ],
            }
        )

    payload = {
        "original_question": question,
        "current_round": round_id,
        "max_rounds": max_rounds,
        "available_tools": [
            "search(query)",
            "get_document_window(docid, keyword)",
        ],
        "stop_conditions": [
            "credible answer found",
            "maximum rounds reached",
            "new tool round brings no new information",
            "all next actions repeat previous actions",
        ],
        "previous_actions": previous_actions,
        "confirmed_key_facts": confirmed_facts,
        "older_rounds_summary": summarize_window_rounds(older),
        "recent_tool_rounds": recent_payload,
    }
    return [
        {"role": "system", "content": BASIC_WITH_WINDOW_LOOP_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_window_final_messages(
    question: str,
    status: str,
    tool_rounds: List[Dict[str, Any]],
    confirmed_facts: List[str],
    candidate_answers: List[Dict[str, Any]],
    previous_actions: List[Dict[str, str]],
    recent_rounds: int,
    context_snippet_chars: int,
) -> List[Dict[str, str]]:
    older = tool_rounds[:-recent_rounds] if recent_rounds > 0 else tool_rounds
    recent = tool_rounds[-recent_rounds:] if recent_rounds > 0 else []
    recent_payload = []
    for item in recent:
        recent_payload.append(
            {
                "round": item.get("round"),
                "actions": item.get("actions", []),
                "new_docids": item.get("new_docids", []),
                "documents": [
                    compact_tool_result(result, context_snippet_chars)
                    for result in item.get("results", [])
                ],
                "decision": item.get("decision", {}),
            }
        )

    payload = {
        "original_question": question,
        "stop_status": status,
        "previous_actions": previous_actions,
        "confirmed_key_facts": confirmed_facts,
        "candidate_answers": candidate_answers[-8:],
        "older_rounds_summary": summarize_window_rounds(older),
        "recent_tool_rounds": recent_payload,
    }
    return [
        {"role": "system", "content": BASIC_WITH_WINDOW_FINAL_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def execute_basic_window_action(
    action: Dict[str, str],
    tool_registry: Dict[str, Callable[..., Any]],
) -> List[Dict[str, Any]]:
    tool_name = action.get("tool", "")
    if tool_name == "search":
        return tool_registry["search"](query=action["query"])
    if tool_name == "get_document_window":
        return tool_registry["get_document_window"](docid=action["docid"], keyword=action["keyword"])
    return [{"docid": "", "found": False, "error": f"unknown tool: {tool_name}", "snippet": ""}]


def collect_new_information(
    results: List[Dict[str, Any]],
    seen_docids: set,
    seen_result_keys: set,
) -> Tuple[List[str], List[str]]:
    new_docids: List[str] = []
    new_result_keys: List[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("found") is False and not result.get("snippet"):
            continue
        docid = str(result.get("docid", "")).strip()
        key = result_key(result)
        if docid and docid not in seen_docids:
            new_docids.append(docid)
        if key and key not in seen_result_keys:
            new_result_keys.append(key)
        if docid:
            seen_docids.add(docid)
        if key:
            seen_result_keys.add(key)
    return new_docids, new_result_keys


def run_basic_with_window_agent(
    client: Any,
    model: str,
    question: str,
    searcher: Any,
    top_k: int = 6,
    max_rounds: int = 10,
    max_tokens: int = 4096,
    snippet_max_chars: int = 1600,
    window_chars: int = 1200,
    recent_rounds: int = 3,
    context_snippet_chars: int = 1200,
) -> Dict[str, Any]:
    _search_specs, search_registry = get_basic_tool_specs_and_registry(
        searcher=searcher,
        k=top_k,
        snippet_max_chars=snippet_max_chars,
    )
    _window_specs, window_registry = get_document_window_tool_specs_and_registry(
        searcher=searcher,
        window_chars=window_chars,
    )
    tool_registry = {**search_registry, **window_registry}

    transcript_messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a Basic Deep Research Agent with an additional document-window tool "
                "for an ablation experiment."
            ),
        },
        {"role": "user", "content": question},
    ]

    tool_rounds: List[Dict[str, Any]] = []
    confirmed_facts: List[str] = []
    candidate_answers: List[Dict[str, Any]] = []
    previous_actions: List[Dict[str, str]] = []
    previous_signatures = set()
    seen_docids = set()
    seen_result_keys = set()
    pending_actions: List[Dict[str, str]] = [{"tool": "search", "query": normalize_query(question)}]
    final_output = ""
    status = "max_rounds_reached"
    last_decision = normalize_window_decision(None)

    for round_id in range(1, max_rounds + 1):
        deduped_actions: List[Dict[str, str]] = []
        local_seen = set()
        for action in pending_actions:
            signature = action_signature(action)
            if not signature or signature in previous_signatures or signature in local_seen:
                continue
            deduped_actions.append(action)
            local_seen.add(signature)

        if not deduped_actions:
            status = "repeated_action" if pending_actions else "no_next_action"
            break

        tool_calls = []
        round_results: List[Dict[str, Any]] = []
        for action_index, action in enumerate(deduped_actions, start=1):
            call_id = f"bw_{round_id}_{action_index}"
            args = (
                {"query": action["query"]}
                if action["tool"] == "search"
                else {"docid": action["docid"], "keyword": action["keyword"]}
            )
            tool_calls.append(make_tool_call(call_id, action["tool"], args))

        transcript_messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})

        for action, tool_call in zip(deduped_actions, tool_calls):
            previous_actions.append(action)
            previous_signatures.add(action_signature(action))
            results = execute_basic_window_action(action, tool_registry)
            round_results.extend(results)
            transcript_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(results, ensure_ascii=False),
                }
            )

        new_docids, new_result_keys = collect_new_information(
            results=round_results,
            seen_docids=seen_docids,
            seen_result_keys=seen_result_keys,
        )
        new_information = bool(new_docids or new_result_keys)

        round_record: Dict[str, Any] = {
            "round": round_id,
            "actions": deduped_actions,
            "results": round_results,
            "new_docids": new_docids,
            "new_information": new_information,
        }
        tool_rounds.append(round_record)

        decision_messages = build_window_decision_messages(
            question=question,
            tool_rounds=tool_rounds,
            confirmed_facts=confirmed_facts,
            previous_actions=previous_actions,
            round_id=round_id,
            max_rounds=max_rounds,
            recent_rounds=recent_rounds,
            context_snippet_chars=context_snippet_chars,
        )
        response = client.simple_chat(
            model=model,
            messages=decision_messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw_content = response["choices"][0]["message"].get("content", "")
        parsed = extract_json_object(raw_content)
        decision = normalize_window_decision(parsed)
        last_decision = decision
        round_record["decision"] = decision
        round_record["raw_decision"] = raw_content

        transcript_messages.extend(decision_messages)
        transcript_messages.append({"role": "assistant", "content": raw_content})

        add_confirmed_facts(confirmed_facts, decision["key_facts"])
        if decision["final_answer"]:
            candidate_answers.append(
                {
                    "answer": decision["final_answer"],
                    "confidence": decision["confidence"],
                    "evidence_sufficient": decision["evidence_sufficient"],
                    "used_docids": decision["used_docids"],
                    "round": round_id,
                }
            )

        if decision["evidence_sufficient"] and decision["final_answer"]:
            status = "completed"
            basic_decision = {
                "analysis": decision["analysis"],
                "final_answer": decision["final_answer"],
                "confidence": decision["confidence"],
                "used_docids": decision["used_docids"],
            }
            final_output = format_basic_final_answer(basic_decision, status)
            transcript_messages[-1]["content"] = final_output
            break

        if not new_information and round_id > 1:
            status = "no_new_information"
            break

        pending_actions = decision["actions"]
        if not pending_actions:
            status = "no_next_action"
            break

    if not final_output:
        final_messages = build_window_final_messages(
            question=question,
            status=status,
            tool_rounds=tool_rounds,
            confirmed_facts=confirmed_facts,
            candidate_answers=candidate_answers,
            previous_actions=previous_actions,
            recent_rounds=recent_rounds,
            context_snippet_chars=context_snippet_chars,
        )
        response = client.simple_chat(
            model=model,
            messages=final_messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw_final_content = response["choices"][0]["message"].get("content", "")
        parsed_final = extract_json_object(raw_final_content)
        final_decision = normalize_window_decision(parsed_final)
        if not final_decision["final_answer"] and candidate_answers:
            candidate = candidate_answers[-1]
            final_decision = normalize_window_decision(
                {
                    "analysis": (
                        "The tool loop stopped before full verification, so this is the most "
                        "plausible candidate from the accumulated evidence."
                    ),
                    "final_answer": candidate.get("answer", ""),
                    "confidence": candidate.get("confidence", "low"),
                    "used_docids": candidate.get("used_docids", []),
                    "key_facts": [],
                    "actions": [],
                }
            )
        if not final_decision["final_answer"]:
            final_decision = last_decision
        basic_final_decision = {
            "analysis": final_decision["analysis"],
            "final_answer": final_decision["final_answer"],
            "confidence": final_decision["confidence"],
            "used_docids": final_decision["used_docids"],
        }
        final_output = format_basic_final_answer(basic_final_decision, status)
        transcript_messages.extend(final_messages)
        transcript_messages.append({"role": "assistant", "content": final_output})

    return {
        "query": question,
        "status": status,
        "final_output": final_output,
        "messages": transcript_messages,
        "search_rounds": tool_rounds,
        "confirmed_facts": confirmed_facts,
        "candidate_answers": candidate_answers,
    }


def run_open_track_no_window(
    client: Any,
    model: str,
    question: str,
    searcher: Any,
    top_k: int,
    snippet_max_chars: int,
    max_rounds: int,
    max_tokens: int,
) -> Dict[str, Any]:
    tool_specs, tool_registry = get_search_tool_specs_and_registry(
        searcher=searcher,
        k=top_k,
        snippet_max_chars=snippet_max_chars,
    )
    with patched_open_track_loop_prompt(OPEN_TRACK_NO_WINDOW_LOOP_PROMPT):
        return open_track_agent.run_research_agent(
            client=client,
            model=model,
            query=question,
            tool_specs=tool_specs,
            tool_registry=tool_registry,
            max_rounds=max_rounds,
            max_tokens=max_tokens,
        )


def build_error_record(variant: str, row: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
    return {
        "agent": variant,
        "query_id": row.get("query_id", ""),
        "query": row.get("query", ""),
        "gold_answer": row.get("answer", ""),
        "status": "error",
        "predicted_answer": "",
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "messages": [
            {"role": "system", "content": "Ablation-agent run failed."},
            {"role": "user", "content": row.get("query", "")},
            {"role": "assistant", "content": f"ERROR: {repr(exc)}"},
        ],
    }


def run_variant_prediction(
    variant: str,
    args: argparse.Namespace,
    client: VLLMClient,
    searcher: Any,
    question: str,
) -> Dict[str, Any]:
    if variant == "open_track_no_window":
        return run_open_track_no_window(
            client=client,
            model=args.model,
            question=question,
            searcher=searcher,
            top_k=args.top_k,
            snippet_max_chars=args.snippet_max_chars,
            max_rounds=args.max_rounds,
            max_tokens=args.max_tokens,
        )
    if variant == "basic_with_window":
        return run_basic_with_window_agent(
            client=client,
            model=args.model,
            question=question,
            searcher=searcher,
            top_k=args.top_k,
            max_rounds=args.max_rounds,
            max_tokens=args.max_tokens,
            snippet_max_chars=args.snippet_max_chars,
            window_chars=args.window_chars,
            recent_rounds=args.recent_rounds,
            context_snippet_chars=args.context_snippet_chars,
        )
    raise ValueError(f"Unknown ablation variant: {variant}")


def run_predictions_for_variant(
    variant: str,
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    submission_path: Path,
) -> None:
    client = VLLMClient(base_url=args.base_url, api_key=args.api_key)
    searcher = build_searcher(index_path=str(resolve_path(args.index_path)))

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    with submission_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows, start=1):
            query_id = str(row.get("query_id", ""))
            print(f"[{variant} {idx:02d}/{len(rows):02d}] query_id={query_id}")
            try:
                result = run_variant_prediction(
                    variant=variant,
                    args=args,
                    client=client,
                    searcher=searcher,
                    question=row["query"],
                )
                record = {
                    "agent": variant,
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "gold_answer": row.get("answer", ""),
                    "status": result.get("status", ""),
                    "predicted_answer": result.get("final_output", ""),
                    "messages": result.get("messages", []),
                }
                for key in ("search_rounds", "confirmed_facts", "candidate_answers"):
                    if key in result:
                        record[key] = result[key]
            except Exception as exc:
                print(f"  ERROR: {repr(exc)}")
                record = build_error_record(variant, row, exc)

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            preview = str(record.get("predicted_answer", "")).replace("\n", " ")[:180]
            print(f"  status={record.get('status')} pred={preview}")


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                records.append(json.loads(line))
    return records


def status_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(record.get("status", "")) for record in records))


def tool_call_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for message in record.get("messages", []):
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                name = (tool_call.get("function") or {}).get("name", "")
                if name:
                    counts[str(name)] += 1
    return dict(counts)


def build_eval_detail_map(details: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(detail.get("query_id", "")): detail for detail in details}


def write_ablation_file(
    ablation_path: Path,
    args: argparse.Namespace,
    timestamp: str,
    dataset_path: Path,
    variant_outputs: Dict[str, Dict[str, Any]],
) -> None:
    payload: Dict[str, Any] = {
        "type": "ablation",
        "timestamp": timestamp,
        "dataset_path": str(dataset_path),
        "variants_requested": list(args.variants),
        "config": {
            "model": args.model,
            "eval_model": args.eval_model or args.model,
            "top_k": args.top_k,
            "snippet_max_chars": args.snippet_max_chars,
            "window_chars": args.window_chars,
            "max_rounds": args.max_rounds,
            "max_tokens": args.max_tokens,
            "eval_max_tokens": args.eval_max_tokens,
            "recent_rounds": args.recent_rounds,
            "context_snippet_chars": args.context_snippet_chars,
            "limit": args.limit,
            "compact": args.compact,
            "skip_eval": args.skip_eval,
        },
        "variants": {},
    }

    for variant, output in variant_outputs.items():
        records = output["submission_records"]
        variant_payload: Dict[str, Any] = {
            "submission_path": str(output["submission_path"]),
            "eval_path": str(output.get("eval_path", "")),
            "summary": output.get("summary", {}),
            "status_counts": status_counts(records),
            "tool_call_counts": tool_call_counts(records),
            "eval_details": output.get("eval_details", []),
        }
        if args.compact:
            variant_payload["submission_preview"] = [
                {
                    "query_id": record.get("query_id"),
                    "status": record.get("status"),
                    "gold_answer": record.get("gold_answer"),
                    "predicted_answer": record.get("predicted_answer"),
                }
                for record in records
            ]
        else:
            variant_payload["submissions"] = records
        payload["variants"][variant] = variant_payload

    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def print_variant_summary(variant: str, summary: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print(f"Ablation variant: {variant}")
    print("=" * 70)
    if summary:
        print(f"total_queries: {summary.get('total_queries')}")
        print(f"correct:       {summary.get('correct')}")
        print(f"incorrect:     {summary.get('incorrect')}")
        print(f"accuracy:      {summary.get('accuracy', 0.0):.2%}")
        print(f"avg_tools:     {summary.get('avg_tool_calls_per_query')}")
        print(f"avg_docs:      {summary.get('avg_retrieved_docs_per_query')}")
    print(f"status_counts: {status_counts(records)}")
    print(f"tool_counts:   {tool_call_counts(records)}")


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    output_dir = resolve_path(args.output_dir)
    timestamp = args.timestamp or datetime.now().strftime("%m%d_%H%M")
    ablation_path = output_dir / f"ablation_{timestamp}.json"

    rows = load_jsonl(dataset_path, limit=args.limit)
    if args.limit == 50 and len(rows) != 50:
        raise ValueError(f"Expected 50 rows from {dataset_path}, got {len(rows)}.")

    variant_outputs: Dict[str, Dict[str, Any]] = {}
    for variant in args.variants:
        submission_path = output_dir / f"ablation_{timestamp}_{variant}_submission.jsonl"
        eval_path = output_dir / f"ablation_{timestamp}_{variant}_eval.jsonl"

        run_predictions_for_variant(
            variant=variant,
            args=args,
            rows=rows,
            submission_path=submission_path,
        )
        records = load_jsonl_records(submission_path)

        summary: Dict[str, Any] = {}
        details: List[Dict[str, Any]] = []
        if not args.skip_eval:
            summary, details = run_evaluation(
                submission_path=str(submission_path),
                dataset_path=str(dataset_path),
                model_name=args.eval_model or args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                output_path=str(eval_path),
                max_tokens=4096,
                verbose=True,
            )

        variant_outputs[variant] = {
            "submission_path": submission_path,
            "eval_path": eval_path if not args.skip_eval else "",
            "summary": summary,
            "eval_details": details,
            "eval_detail_map": build_eval_detail_map(details),
            "submission_records": records,
        }
        print_variant_summary(variant, summary, records)

    write_ablation_file(
        ablation_path=ablation_path,
        args=args,
        timestamp=timestamp,
        dataset_path=dataset_path,
        variant_outputs=variant_outputs,
    )
    print(f"\nCombined ablation file saved to: {ablation_path}")


if __name__ == "__main__":
    main()
