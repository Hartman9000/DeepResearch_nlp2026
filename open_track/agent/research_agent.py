import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from .model_io import append_agent_exchange, call_json_agent, strip_thinking
from .normalization import normalize_anchor_queries, normalize_constraints, normalize_status
from .prompts import EXTRACT_EVIDENCE_PROMPT, LOOP_PROMPT, PARSE_PROMPT
from .tooling import compact_snippet, execute_tool_call, merge_tool_result, run_search


def parse_question_with_model(
    client: Any,
    model: str,
    query: str,
    max_tokens: int = 1600,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], str]:
    user_content = f"Question:\n{query}"
    parsed, raw = call_json_agent(
        client=client,
        model=model,
        system_prompt=PARSE_PROMPT,
        user_content=user_content,
        max_tokens=max_tokens,
    )
    if messages is not None:
        append_agent_exchange(messages, PARSE_PROMPT, user_content, raw)
    target = parsed.get("target")
    if not isinstance(target, dict):
        raise ValueError("parse_agent output must include a valid target.")
    state_plan = {
        "target": {
            "answer_type": str(target.get("answer_type") or "unknown"),
            "description": str(target.get("description") or query),
        },
        "constraints": normalize_constraints(parsed.get("constraints")),
        "anchor_queries": normalize_anchor_queries(parsed.get("anchor_queries"), original_query=query),
    }
    return state_plan, raw


def build_extract_payload(
    state: Dict[str, Any],
    snippet_entrys: List[Dict[str, Any]],
    max_snippets: int = 14,
) -> Dict[str, Any]:
    ranked = sorted(
        snippet_entrys,
        key=lambda item: (
            -int(item.get("retrieval_count", 0)),
            -float(item.get("best_score", 0.0) or 0.0),
            int(item.get("best_rank", 999)),
        ),
    )
    return {
        "original_question": state["original_query"],
        "target": state["target"],
        "constraints": state["constraints"],
        "searched_queries": state["searched_queries"],
        "tool_history": state["tool_history"],
        "analysis_log": state["analysis_log"],
        "current_candidates": state["candidate_answers"][-5:],
        "snippets": [compact_snippet(item) for item in ranked[:max_snippets]],
    }


def extract_evidence_with_model(
    client: Any,
    model: str,
    state: Dict[str, Any],
    snippet_entrys: List[Dict[str, Any]],
    max_tokens: int = 2200,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], str]:
    payload = build_extract_payload(state, snippet_entrys)
    user_content = json.dumps(payload, ensure_ascii=False, indent=2)
    parsed, raw = call_json_agent(
        client=client,
        model=model,
        system_prompt=EXTRACT_EVIDENCE_PROMPT,
        user_content=user_content,
        max_tokens=max_tokens,
    )
    if messages is not None:
        append_agent_exchange(messages, EXTRACT_EVIDENCE_PROMPT, user_content, raw)
    return parsed, raw


def update_state_from_extraction(
    state: Dict[str, Any],
    extraction: Dict[str, Any],
    snippet_bank: Dict[str, Dict[str, Any]],
) -> None:
    by_constraint_id = {item["id"]: item for item in state["constraints"]}
    evidence_by_docid = {item["docid"]: item for item in state["evident_snippets"]}

    for item in extraction.get("selected_snippets", []):
        if not isinstance(item, dict):
            continue
        docid = str(item.get("docid", ""))
        if docid not in snippet_bank:
            continue
        entry = compact_snippet(snippet_bank[docid])
        entry["why"] = str(item.get("why", "")).strip()
        if docid in evidence_by_docid:
            old = evidence_by_docid[docid]
            if entry["why"] and entry["why"] not in old.get("why", ""):
                old["why"] = (old.get("why", "") + " " + entry["why"]).strip()
            old["retrieval_count"] = entry["retrieval_count"]
            old["source_queries"] = entry["source_queries"]
        else:
            state["evident_snippets"].append(entry)
            evidence_by_docid[docid] = entry

    for update in extraction.get("constraint_updates", []):
        if not isinstance(update, dict):
            continue
        constraint_id = str(update.get("id", ""))
        if constraint_id not in by_constraint_id:
            continue
        status = normalize_status(update.get("status"))
        constraint = by_constraint_id[constraint_id]
        if status in {"supported", "contradicted"} or constraint["status"] == "unknown":
            constraint["status"] = status
        docids = [
            str(docid)
            for docid in update.get("evidence_docids", [])
            if str(docid) in snippet_bank
        ]
        if docids:
            constraint["evidence_docids"] = list(dict.fromkeys(constraint.get("evidence_docids", []) + docids))[:5]
        rationale = str(update.get("rationale", "")).strip()
        if rationale:
            constraint["rationale"] = rationale

    for candidate in extraction.get("candidate_answers", []):
        if not isinstance(candidate, dict):
            continue
        answer = str(candidate.get("answer", "")).strip()
        docids = [
            str(docid)
            for docid in candidate.get("evidence_docids", [])
            if str(docid) in snippet_bank
        ]
        if not answer or not docids:
            continue
        key = answer.lower()
        existing = next((item for item in state["candidate_answers"] if item["answer"].lower() == key), None)
        if existing:
            existing["evidence_docids"] = list(dict.fromkeys(existing["evidence_docids"] + docids))[:5]
            existing["confidence"] = max_confidence(existing.get("confidence", "low"), candidate.get("confidence", "low"))
            rationale = str(candidate.get("rationale", "")).strip()
            if rationale and rationale not in existing.get("rationale", ""):
                existing["rationale"] = (existing.get("rationale", "") + " " + rationale).strip()
        else:
            state["candidate_answers"].append(
                {
                    "answer": answer,
                    "confidence": normalize_confidence(candidate.get("confidence", "low")),
                    "evidence_docids": docids[:5],
                    "rationale": str(candidate.get("rationale", "")).strip(),
                }
            )

    analysis = str(extraction.get("analysis", "")).strip()
    if analysis:
        state["analysis_log"].append(analysis)


def normalize_confidence(value: Any) -> str:
    lowered = str(value).strip().lower()
    if lowered in {"high", "medium", "low"}:
        return lowered
    return "low"


def max_confidence(left: str, right: Any) -> str:
    order = {"low": 1, "medium": 2, "high": 3}
    right_norm = normalize_confidence(right)
    return left if order.get(left, 1) >= order.get(right_norm, 1) else right_norm


def constraint_summary(state: Dict[str, Any]) -> Dict[str, int]:
    counts = {"critical": 0, "critical_supported": 0, "strong": 0, "strong_supported": 0, "contradicted": 0}
    for item in state["constraints"]:
        priority = item.get("priority", "strong")
        status = item.get("status", "unknown")
        if priority == "critical":
            counts["critical"] += 1
            if status == "supported":
                counts["critical_supported"] += 1
        if priority == "strong":
            counts["strong"] += 1
            if status == "supported":
                counts["strong_supported"] += 1
        if status == "contradicted":
            counts["contradicted"] += 1
    return counts


def stop_condition(state: Dict[str, Any]) -> Dict[str, Any]:
    counts = constraint_summary(state)
    critical_ok = counts["critical"] == counts["critical_supported"]
    if counts["strong"] == 0:
        strong_ok = True
        strong_ratio = 1.0
    else:
        strong_ratio = counts["strong_supported"] / counts["strong"]
        strong_ok = strong_ratio >= 0.6
    has_candidate = any(item.get("answer") and item.get("evidence_docids") for item in state["candidate_answers"])
    no_contradiction = counts["contradicted"] == 0
    return {
        "passed": bool(has_candidate and critical_ok and strong_ok and no_contradiction),
        "candidate_answer_direct_evidence": bool(has_candidate),
        "critical_constraints_satisfied": bool(critical_ok),
        "strong_constraints_mostly_satisfied": bool(strong_ok),
        "strong_supported_ratio": round(strong_ratio, 3),
        "no_contradiction": bool(no_contradiction),
    }


def build_work_message(state: Dict[str, Any]) -> str:
    evidence = sorted(
        state["evident_snippets"],
        key=lambda item: (-int(item.get("relevance", 1)), -int(item.get("retrieval_count", 0))),
    )
    work_state = {
        "original_question": state["original_query"],
        "target": state["target"],
        "constraints": state["constraints"],
        "candidate_answers": state["candidate_answers"][-6:],
        "evident_snippets": evidence,
        "searched_queries": state["searched_queries"][-12:],
        "tool_history": state["tool_history"][-12:],
        "stop_condition": stop_condition(state),
        "analysis_log": state["analysis_log"]
    }
    return json.dumps(work_state, ensure_ascii=False, indent=2)


def build_loop_messages(state: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": LOOP_PROMPT},
        {"role": "user", "content": build_work_message(state)},
    ]


def format_final_answer(answer: str, evidence_docids: List[str], state: Dict[str, Any]) -> str:
    answer = answer.strip()
    if not answer and state["candidate_answers"]:
        answer = state["candidate_answers"][-1]["answer"]
    docids = list(dict.fromkeys([str(docid) for docid in evidence_docids if docid]))
    if not docids and state["candidate_answers"]:
        docids = state["candidate_answers"][-1].get("evidence_docids", [])
    evidence_line = ", ".join(docids) if docids else "no docid"
    return f"Explanation: Key evidence comes from docid {evidence_line}.\nExact Answer: {answer}\nConfidence: 70%"


def initialize_research_state(
    query: str,
    client: Any,
    model: str,
    max_tokens: int = 1600,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    plan, raw_plan = parse_question_with_model(
        client=client,
        model=model,
        query=query,
        max_tokens=max_tokens,
        messages=messages,
    )
    return {
        "original_query": query,
        "target": plan["target"],
        "constraints": plan["constraints"],
        "anchor_queries": plan["anchor_queries"],
        "searched_queries": [],
        "snippet_bank": {},
        "evident_snippets": [],
        "candidate_answers": [],
        "tool_history": [],
        "analysis_log": [],
        "raw_outputs": {"parse": raw_plan, "extract": [], "loop": []},
    }


def run_research_agent(
    client: Any,
    model: str,
    query: str,
    tool_specs: List[Dict[str, Any]],
    tool_registry: Dict[str, Callable[..., Any]],
    max_rounds: int = 8,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    transcript_messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a Deep Research Agent. Use search and document-window tools to collect evidence step by step and answer the question in English.",
        },
        {"role": "user", "content": query},
    ]
    state = initialize_research_state(
        query=query,
        client=client,
        model=model,
        max_tokens=max_tokens,
        messages=transcript_messages,
    )
    snippet_bank: Dict[str, Dict[str, Any]] = state["snippet_bank"] # {docid:entry}

    initial_queries = state["anchor_queries"]
    for idx, search_query in enumerate(initial_queries, start=1):
        call_id = f"anchor_search_{idx}"
        tool_call, executed = run_search(search_query, tool_registry, call_id)
        search_results = executed.get("tool_result", []) # List[RetrievedDocument]
        transcript_messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        transcript_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(search_results, ensure_ascii=False),
            }
        )
        merge_tool_result(state, snippet_bank, executed)
    if not snippet_bank:
        raise ValueError("Initial anchor queries returned no search results.")

    extraction, raw = extract_evidence_with_model(
        client=client,
        model=model,
        state=state,
        snippet_entrys=list(snippet_bank.values()),
        max_tokens=max_tokens,
        messages=transcript_messages,
    )
    state["raw_outputs"]["extract"].append(raw)

    update_state_from_extraction(state, extraction, snippet_bank)

    final_output = ""
    status = "max_rounds_reached"

    for round_id in range(1, max_rounds + 1):
        loop_messages = build_loop_messages(state)
        response = client.simple_chat(
            model=model,
            messages=loop_messages,
            temperature=0.0,
            max_tokens=max_tokens,
            tools=tool_specs,
            tool_choice="auto",
        )
        message = response["choices"][0]["message"]
        raw_content = str(message.get("content") or "")
        visible_content = strip_thinking(raw_content)
        tool_calls = message.get("tool_calls") or []
        state["raw_outputs"]["loop"].append(raw_content)
        if visible_content:
            state["analysis_log"].append(visible_content)

        transcript_messages.extend(loop_messages)
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": raw_content}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        transcript_messages.append(assistant_message)

        if not tool_calls:
            current_stop = stop_condition(state)
            if current_stop["passed"]:
                if "Exact Answer" in visible_content:
                    final_output = visible_content
                elif state["candidate_answers"]:
                    candidate = state["candidate_answers"][-1]
                    final_output = format_final_answer(
                        answer=candidate["answer"],
                        evidence_docids=candidate.get("evidence_docids", []),
                        state=state,
                    )
                else:
                    final_output = visible_content
                status = "completed"
            else:
                final_output = visible_content
                status = "stopped_without_tool_calls"
            if status == "completed" and final_output:
                transcript_messages[-1]["content"] = final_output
            break

        round_touched: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            executed = execute_tool_call(tool_call, tool_registry)
            transcript_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(executed["tool_result"], ensure_ascii=False),
                }
            )
            round_touched.extend(merge_tool_result(state, snippet_bank, executed))

        if round_touched:
            extraction, raw_extract = extract_evidence_with_model(
                client=client,
                model=model,
                state=state,
                snippet_entrys=round_touched,
                max_tokens=max_tokens,
                messages=transcript_messages,
            )
            state["raw_outputs"]["extract"].append(raw_extract)
            update_state_from_extraction(state, extraction, snippet_bank)

    else:
        if state["candidate_answers"]:
            candidate = state["candidate_answers"][-1]
            final_output = format_final_answer(
                answer=candidate["answer"],
                evidence_docids=candidate.get("evidence_docids", []),
                state=state,
            )
            transcript_messages.append({"role": "assistant", "content": final_output})
            status = "max_rounds_with_candidate"

    return {
        "query": query,
        "status": status,
        "final_output": final_output,
        "messages": transcript_messages,
    }
