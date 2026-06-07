import json
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .prompts import BASIC_FINAL_SYSTEM_PROMPT, BASIC_LOOP_SYSTEM_PROMPT


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    visible = strip_thinking(text)
    candidates = [visible, text.strip()]

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", visible, flags=re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))

    first = visible.find("{")
    last = visible.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.insert(0, visible[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_query(query: Any) -> str:
    return " ".join(str(query or "").split()).strip(" \t\r\n\"'")


def normalize_confidence(value: Any) -> str:
    value = str(value or "").strip().lower()
    if value in {"low", "medium", "high"}:
        return value
    return "low"


def normalize_decision(raw_decision: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    decision = raw_decision if isinstance(raw_decision, dict) else {}
    key_facts = decision.get("key_facts", [])
    if not isinstance(key_facts, list):
        key_facts = []
    used_docids = decision.get("used_docids", [])
    if not isinstance(used_docids, list):
        used_docids = []
    return {
        "analysis": str(decision.get("analysis", "")).strip(),
        "evidence_sufficient": bool(decision.get("evidence_sufficient", False)),
        "final_answer": str(decision.get("final_answer", "")).strip(),
        "confidence": normalize_confidence(decision.get("confidence")),
        "used_docids": [str(docid) for docid in used_docids if str(docid).strip()],
        "key_facts": [str(fact).strip() for fact in key_facts if str(fact).strip()],
        "next_query": normalize_query(decision.get("next_query", "")),
    }


def make_search_tool_call(call_id: str, query: str) -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "search",
            "arguments": json.dumps({"query": query}, ensure_ascii=False),
        },
    }


def compact_result(result: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    snippet = str(result.get("snippet", ""))
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "..."
    return {
        "docid": str(result.get("docid", "")),
        "score": result.get("score", 0.0),
        "url": result.get("url", ""),
        "snippet": snippet,
    }


def result_key(result: Dict[str, Any]) -> str:
    docid = str(result.get("docid", ""))
    snippet = " ".join(str(result.get("snippet", "")).split())
    return f"{docid}:{snippet[:500]}"


def add_confirmed_facts(confirmed_facts: List[str], new_facts: List[str], limit: int = 24) -> None:
    seen = {fact.lower() for fact in confirmed_facts}
    for fact in new_facts:
        key = fact.lower()
        if key and key not in seen:
            confirmed_facts.append(fact)
            seen.add(key)
        if len(confirmed_facts) >= limit:
            break


def summarize_older_rounds(rounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for item in rounds:
        decision = item.get("decision", {})
        summary.append(
            {
                "round": item.get("round"),
                "query": item.get("query"),
                "docids": [str(result.get("docid", "")) for result in item.get("results", [])],
                "new_docids": item.get("new_docids", []),
                "analysis": str(decision.get("analysis", ""))[:500],
                "key_facts": decision.get("key_facts", [])[:5],
            }
        )
    return summary


def build_decision_messages(
    question: str,
    search_rounds: List[Dict[str, Any]],
    confirmed_facts: List[str],
    previous_queries: List[str],
    round_id: int,
    max_rounds: int,
    recent_rounds: int,
    context_snippet_chars: int,
) -> List[Dict[str, str]]:
    older = search_rounds[:-recent_rounds] if recent_rounds > 0 else search_rounds
    recent = search_rounds[-recent_rounds:] if recent_rounds > 0 else []
    recent_payload = []
    for item in recent:
        recent_payload.append(
            {
                "round": item.get("round"),
                "query": item.get("query"),
                "new_docids": item.get("new_docids", []),
                "documents": [
                    compact_result(result, context_snippet_chars)
                    for result in item.get("results", [])
                ],
            }
        )

    payload = {
        "original_question": question,
        "current_round": round_id,
        "max_rounds": max_rounds,
        "available_tool": "search(query)",
        "stop_conditions": [
            "credible answer found",
            "maximum search rounds reached",
            "new search brings no new information",
        ],
        "previous_queries": previous_queries,
        "confirmed_key_facts": confirmed_facts,
        "older_rounds_summary": summarize_older_rounds(older),
        "recent_search_rounds": recent_payload,
    }
    return [
        {"role": "system", "content": BASIC_LOOP_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_final_messages(
    question: str,
    status: str,
    search_rounds: List[Dict[str, Any]],
    confirmed_facts: List[str],
    candidate_answers: List[Dict[str, Any]],
    previous_queries: List[str],
    recent_rounds: int,
    context_snippet_chars: int,
) -> List[Dict[str, str]]:
    older = search_rounds[:-recent_rounds] if recent_rounds > 0 else search_rounds
    recent = search_rounds[-recent_rounds:] if recent_rounds > 0 else []
    recent_payload = []
    for item in recent:
        recent_payload.append(
            {
                "round": item.get("round"),
                "query": item.get("query"),
                "new_docids": item.get("new_docids", []),
                "documents": [
                    compact_result(result, context_snippet_chars)
                    for result in item.get("results", [])
                ],
                "decision": item.get("decision", {}),
            }
        )

    payload = {
        "original_question": question,
        "stop_status": status,
        "previous_queries": previous_queries,
        "confirmed_key_facts": confirmed_facts,
        "candidate_answers": candidate_answers[-8:],
        "older_rounds_summary": summarize_older_rounds(older),
        "recent_search_rounds": recent_payload,
    }
    return [
        {"role": "system", "content": BASIC_FINAL_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def format_final_answer(decision: Dict[str, Any], status: str) -> str:
    answer = str(decision.get("final_answer", "")).strip()
    confidence = normalize_confidence(decision.get("confidence"))
    docids = [str(docid) for docid in decision.get("used_docids", []) if str(docid).strip()]
    analysis = str(decision.get("analysis", "")).strip()
    if not answer:
        answer = "NOT FOUND"
    evidence = ", ".join(docids) if docids else "no direct docid"
    explanation = analysis or f"Stopped with status: {status}."
    return f"Explanation: {explanation} Evidence docids: {evidence}.\nExact Answer: {answer}\nConfidence: {confidence}"


def run_basic_agent(
    client: Any,
    model: str,
    question: str,
    searcher: Any,
    search_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
    top_k: int = 5,
    max_rounds: int = 6,
    max_tokens: int = 2048,
    snippet_max_chars: int = 1600,
    recent_rounds: int = 3,
    context_snippet_chars: int = 1200,
    initial_query: Optional[str] = None,
) -> Dict[str, Any]:
    if search_fn is None:
        from .tools import get_basic_tool_specs_and_registry

        _, registry = get_basic_tool_specs_and_registry(
            searcher=searcher,
            k=top_k,
            snippet_max_chars=snippet_max_chars,
        )
        search_fn = registry["search"]

    transcript_messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a Basic Deep Research Agent. Search iteratively, judge evidence after each "
                "round, rewrite queries when evidence is insufficient, and answer only from evidence."
            ),
        },
        {"role": "user", "content": question},
    ]

    search_rounds: List[Dict[str, Any]] = []
    confirmed_facts: List[str] = []
    candidate_answers: List[Dict[str, Any]] = []
    previous_queries: List[str] = []
    seen_docids: Set[str] = set()
    seen_result_keys: Set[str] = set()
    next_query = normalize_query(initial_query) or normalize_query(question)
    final_output = ""
    status = "max_rounds_reached"
    last_decision = normalize_decision(None)

    for round_id in range(1, max_rounds + 1):
        search_query = normalize_query(next_query)
        if not search_query:
            status = "no_next_query"
            break
        if search_query.lower() in {query.lower() for query in previous_queries}:
            status = "repeated_query"
            break

        call_id = f"search_{round_id}"
        tool_call = make_search_tool_call(call_id, search_query)
        results = search_fn(search_query)
        previous_queries.append(search_query)

        new_docids = []
        new_result_keys = []
        for result in results:
            docid = str(result.get("docid", ""))
            key = result_key(result)
            if docid and docid not in seen_docids:
                new_docids.append(docid)
            if key and key not in seen_result_keys:
                new_result_keys.append(key)
            if docid:
                seen_docids.add(docid)
            if key:
                seen_result_keys.add(key)
        new_information = bool(new_docids or new_result_keys)

        transcript_messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        transcript_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(results, ensure_ascii=False),
            }
        )

        round_record: Dict[str, Any] = {
            "round": round_id,
            "query": search_query,
            "results": results,
            "new_docids": new_docids,
            "new_information": new_information,
        }
        search_rounds.append(round_record)

        decision_messages = build_decision_messages(
            question=question,
            search_rounds=search_rounds,
            confirmed_facts=confirmed_facts,
            previous_queries=previous_queries,
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
        decision = normalize_decision(parsed)
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
            final_output = format_final_answer(decision, status)
            transcript_messages[-1]["content"] = final_output
            break

        if not new_information and round_id > 1:
            status = "no_new_information"
            break

        next_query = decision["next_query"]
        if not next_query:
            status = "no_next_query"
            break

    if not final_output:
        final_messages = build_final_messages(
            question=question,
            status=status,
            search_rounds=search_rounds,
            confirmed_facts=confirmed_facts,
            candidate_answers=candidate_answers,
            previous_queries=previous_queries,
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
        final_decision = normalize_decision(parsed_final)
        if not final_decision["final_answer"] and candidate_answers:
            candidate = candidate_answers[-1]
            final_decision = normalize_decision(
                {
                    "analysis": (
                        "The search loop stopped before full verification, so this is the most "
                        "plausible candidate from the accumulated evidence."
                    ),
                    "final_answer": candidate.get("answer", ""),
                    "confidence": candidate.get("confidence", "low"),
                    "used_docids": candidate.get("used_docids", []),
                    "key_facts": [],
                    "next_query": "",
                }
            )
        if not final_decision["final_answer"]:
            final_decision = last_decision
        final_output = format_final_answer(final_decision, status)
        transcript_messages.extend(final_messages)
        transcript_messages.append({"role": "assistant", "content": final_output})

    return {
        "query": question,
        "status": status,
        "final_output": final_output,
        "messages": transcript_messages,
        "search_rounds": search_rounds,
        "confirmed_facts": confirmed_facts,
        "candidate_answers": candidate_answers,
    }
