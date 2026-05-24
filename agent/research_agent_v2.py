import json
import re
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "by",
    "can",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "inclusive",
    "into",
    "is",
    "it",
    "its",
    "me",
    "of",
    "on",
    "one",
    "or",
    "per",
    "she",
    "some",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "with",
    "worked",
    "would",
    "year",
    "years",
}


def execute_tool_call(tool_call: Dict[str, Any], registry: Dict[str, Callable[..., Any]]) -> Dict[str, Any]:
    function = tool_call.get("function", {})
    name = function.get("name", "")
    arguments = function.get("arguments", "{}")

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if name not in registry:
        raise ValueError(f"Unknown tool: {name}")

    result = registry[name](**arguments)
    return {
        "tool_name": name,
        "arguments": arguments,
        "tool_result": result,
    }


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction for planner outputs that may include prose."""
    text = text.strip()
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [text]
    if fenced:
        candidates.insert(0, fenced.group(1))

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.insert(0, text[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def keyword_tokens(text: str) -> List[str]:
    tokens = re.findall(
        r"[A-Za-z][A-Za-z0-9'\-]*|[$]?\d[\d,]*(?:\.\d+)?(?:s|%)?",
        text.lower(),
    )
    cleaned = []
    seen = set()
    for token in tokens:
        token = token.strip("'-.")
        if len(token) <= 2 and not token.isdigit():
            continue
        if token in STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            cleaned.append(token)
    return cleaned


def make_anchor_query(text: str, max_terms: int = 12) -> str:
    quoted = re.findall(r"['\"]([^'\"]{3,80})['\"]", text)
    terms = []
    seen = set()

    for phrase in quoted:
        phrase = " ".join(keyword_tokens(phrase))
        if phrase and phrase not in seen:
            seen.add(phrase)
            terms.append(phrase)

    for token in keyword_tokens(text):
        if token not in seen:
            seen.add(token)
            terms.append(token)
        if len(terms) >= max_terms:
            break

    return " ".join(terms[:max_terms]) or text[:160]


def heuristic_constraints(query: str) -> List[Dict[str, Any]]:
    normalized = re.sub(r"\s+", " ", query).strip()
    split_pattern = (
        r"(?<=[.!?])\s+|;\s+|\s+(?=Additionally\b|According to\b|In one\b|In this\b|From page\b|"
        r"One of\b|The author\b|This author\b|The pair\b|The librarian\b|The club\b|The company\b)"
    )
    chunks = [chunk.strip(" ,") for chunk in re.split(split_pattern, normalized) if chunk.strip(" ,")]

    constraints: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        constraints.append(
            {
                "id": f"c{idx}",
                "text": chunk,
                "kind": infer_constraint_kind(chunk),
                "priority": 3 if idx <= 3 else 2,
            }
        )

    if not constraints:
        constraints.append(
            {
                "id": "c1",
                "text": normalized,
                "kind": "target",
                "priority": 3,
            }
        )
    return constraints[:12]


def infer_constraint_kind(text: str) -> str:
    lowered = text.lower()
    if "what is" in lowered or "can you tell" in lowered or "name of" in lowered:
        return "target"
    if re.search(r"\b(18|19|20)\d{2}s?\b", lowered):
        return "date"
    if any(word in lowered for word in ("book", "chapter", "published", "author", "title")):
        return "bibliographic"
    if any(word in lowered for word in ("company", "university", "club", "city", "report")):
        return "entity"
    return "relation"


def fallback_plan(query: str) -> Dict[str, Any]:
    constraints = heuristic_constraints(query)
    search_queries = []
    for constraint in constraints:
        anchor = make_anchor_query(constraint["text"])
        if anchor and anchor not in search_queries:
            search_queries.append(anchor)

    return {
        "target": {
            "answer_type": "unknown",
            "final_question": query,
        },
        "constraints": constraints,
        "subquestions": [
            {
                "id": f"sq{idx}",
                "question": constraint["text"],
                "depends_on": [],
                "status": "open",
            }
            for idx, constraint in enumerate(constraints, start=1)
        ],
        "initial_search_queries": search_queries[:4] or [make_anchor_query(query)],
    }


def normalize_plan(plan: Dict[str, Any], query: str) -> Dict[str, Any]:
    fallback = fallback_plan(query)
    target = plan.get("target") if isinstance(plan.get("target"), dict) else fallback["target"]

    raw_constraints = plan.get("constraints")
    if not isinstance(raw_constraints, list) or not raw_constraints:
        raw_constraints = fallback["constraints"]

    constraints = []
    for idx, item in enumerate(raw_constraints[:12], start=1):
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        constraints.append(
            {
                "id": str(item.get("id") or f"c{idx}"),
                "text": text,
                "kind": str(item.get("kind") or infer_constraint_kind(text)),
                "priority": parse_priority(item.get("priority"), 3 if idx <= 3 else 2),
                "status": "unknown",
                "evidence_docids": [],
            }
        )

    if not constraints:
        constraints = deepcopy(fallback["constraints"])
        for item in constraints:
            item["status"] = "unknown"
            item["evidence_docids"] = []

    raw_subquestions = plan.get("subquestions")
    if not isinstance(raw_subquestions, list) or not raw_subquestions:
        raw_subquestions = fallback["subquestions"]

    subquestions = []
    for idx, item in enumerate(raw_subquestions[:12], start=1):
        if isinstance(item, str):
            item = {"question": item}
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        depends_on = item.get("depends_on") if isinstance(item.get("depends_on"), list) else []
        subquestions.append(
            {
                "id": str(item.get("id") or f"sq{idx}"),
                "question": question,
                "depends_on": [str(dep) for dep in depends_on],
                "status": str(item.get("status") or "open"),
            }
        )

    raw_queries = plan.get("initial_search_queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        raw_queries = fallback["initial_search_queries"]

    initial_queries = []
    for item in raw_queries:
        query_text = make_anchor_query(str(item), max_terms=12)
        if query_text and query_text not in initial_queries:
            initial_queries.append(query_text)

    if not initial_queries:
        initial_queries = [make_anchor_query(query)]

    return {
        "target": {
            "answer_type": str(target.get("answer_type") or "unknown"),
            "final_question": str(target.get("final_question") or query),
        },
        "constraints": constraints,
        "subquestions": subquestions,
        "initial_search_queries": initial_queries[:4],
    }


def parse_priority(value: Any, default: int) -> int:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"high", "critical", "must"}:
            return 3
        if lowered in {"medium", "normal"}:
            return 2
        if lowered in {"low", "minor"}:
            return 1
    try:
        priority = int(value)
    except (TypeError, ValueError):
        priority = default
    return max(1, min(3, priority))


def plan_question_with_model(client: Any, model: str, query: str, max_tokens: int = 1200) -> Tuple[Dict[str, Any], str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You initialize a deep research state for hard benchmark questions. "
                "Return strict JSON only. Do not answer the question. "
                "Your job is to parse the question into a stable verification checklist "
                "and an initial search plan for a BM25-only research agent. "
                "Break the question into atomic constraints, dynamic subquestions, "
                "and short first-round BM25 search queries. "
                "Preserve exact names, numbers, dates, page ranges, titles, institutions, "
                "and quoted phrases exactly as written."
            ),
        },
        {
            "role": "user",
            "content": (
                "Question:\n"
                f"{query}\n\n"
                "Return this JSON schema:\n"
                "{\n"
                '  "target": {"answer_type": "...", "final_question": "..."},\n'
                '  "constraints": [\n'
                '    {"id": "c1", "text": "...", "kind": "entity|date|relation|bibliographic|target", "priority": 3}\n'
                "  ],\n"
                '  "subquestions": [\n'
                '    {"id": "sq1", "question": "...", "depends_on": ["c1"]}\n'
                "  ],\n"
                '  "initial_search_queries": ["short rare-term BM25 query", "..."]\n'
                "}\n\n"
                "Definitions:\n"
                "- constraints are the stable checklist used to judge whether a candidate answer is correct. "
                "They must come from the original question and be atomic, explicit, and checkable.\n"
                "- subquestions are temporary research tasks used to find bridge entities, source documents, "
                "or evidence. They guide search and may combine, expand, or operationalize constraints.\n"
                "- initial_search_queries are first-round BM25 searches for anchor discovery, not final-answer guesses.\n\n"
                "Constraint rules:\n"
                "- Each constraint must express one verifiable condition only.\n"
                "- Include a target constraint when the answer must be a person, title, company, date, location, etc.\n"
                "- Use priority 3 for critical constraints required for a correct answer.\n"
                "- Use priority 2 for useful constraints that help distinguish candidates.\n"
                "- Use priority 1 for minor or weakly distinguishing constraints.\n"
                "- Do not merge unrelated facts into one constraint.\n\n"
                "Subquestion rules:\n"
                "- Subquestions should describe what the agent needs to find or verify next.\n"
                "- Subquestions should be actionable for search, bridge finding, or candidate verification.\n"
                "- A subquestion may depend on one or more constraints.\n"
                "- Do not treat subquestions as final correctness criteria; constraints serve that role.\n\n"
                "Initial BM25 query rules:\n"
                "- Generate short, specific queries for the first forced search loop.\n"
                "- Do not search the full question or use full-sentence wording.\n"
                "- Prefer rare phrases, exact quoted text, distinctive names, titles, years, institutions, "
                "document types, page numbers, financial terms, bibliographic terms, and relationship anchors.\n"
                "- Each query should usually contain 2-6 high-signal terms.\n"
                "- Include exact phrases in quotes when the question contains distinctive wording.\n"
                "- Prefer anchor-discovery queries that can reveal bridge entities or source documents.\n"
                "- Avoid generic constraints such as born, married, author, company, or published unless combined "
                "with a rare entity, exact phrase, year, title, institution, or document type.\n"
                "- Include a small variety of query types when possible: rare phrase, source document, "
                "year/entity anchor, relation bridge, and broad fallback.\n"
                "- Do not include queries whose only purpose is to guess the final answer directly."
            ),
        },
    ]
    response = client.simple_chat(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    raw = response["choices"][0]["message"].get("content", "")
    parsed = extract_json_object(raw)
    if parsed is None:
        parsed = fallback_plan(query)
    return parsed, raw


def initialize_research_state(
    query: str,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    planning_max_tokens: int = 1200,
    use_model_planner: bool = True,
) -> Dict[str, Any]:
    raw_plan = ""
    if use_model_planner and client is not None and model:
        try:
            plan, raw_plan = plan_question_with_model(client, model, query, max_tokens=planning_max_tokens)
        except Exception as exc:
            plan = fallback_plan(query)
            raw_plan = f"planner_error: {type(exc).__name__}: {exc}"
    else:
        plan = fallback_plan(query)

    normalized = normalize_plan(plan, query)
    return {
        "original_query": query,
        "target": normalized["target"],
        "constraints": normalized["constraints"],
        "subquestions": normalized["subquestions"],
        "initial_search_queries": normalized["initial_search_queries"],
        "searched_queries": [],
        "evidence_bank": [],
        "candidate_answers": [],
        "gaps": [],
        "rounds": [],
        "planner_raw": raw_plan,
        "stop_checks": [],
    }


def update_state_with_search_results(state: Dict[str, Any], search_query: str, results: List[Dict[str, Any]]) -> None:
    state["searched_queries"].append(search_query)
    existing_docids = {item["docid"] for item in state["evidence_bank"]}

    for rank, result in enumerate(results, start=1):
        docid = str(result.get("docid", ""))
        if not docid:
            continue
        if docid not in existing_docids:
            state["evidence_bank"].append(
                {
                    "docid": docid,
                    "url": result.get("url", ""),
                    "score": result.get("score", 0.0),
                    "snippet": result.get("snippet", ""),
                    "source_query": search_query,
                    "rank": rank,
                }
            )
            existing_docids.add(docid)

    refresh_constraint_status(state)


def keyword_overlap_score(left: str, right: str) -> int:
    left_terms = set(keyword_tokens(left))
    right_terms = set(keyword_tokens(right))
    if not left_terms or not right_terms:
        return 0
    return len(left_terms & right_terms)


def refresh_constraint_status(state: Dict[str, Any]) -> None:
    for constraint in state["constraints"]:
        evidence_docids = []
        for evidence in state["evidence_bank"]:
            score = keyword_overlap_score(constraint["text"], evidence.get("snippet", ""))
            if score >= 2:
                evidence_docids.append(evidence["docid"])
        constraint["evidence_docids"] = evidence_docids[:5]
        if evidence_docids:
            constraint["status"] = "candidate_supported"
        elif constraint.get("status") != "supported":
            constraint["status"] = "unknown"

    gaps = []
    for constraint in state["constraints"]:
        if constraint["status"] == "unknown" and constraint.get("priority", 1) >= 2:
            gaps.append(
                {
                    "constraint_id": constraint["id"],
                    "text": constraint["text"],
                    "priority": constraint.get("priority", 1),
                    "suggested_query": make_anchor_query(constraint["text"]),
                }
            )
    state["gaps"] = gaps[:6]


def compact_state_for_prompt(state: Dict[str, Any], max_evidence: int = 8) -> str:
    compact = {
        "target": state["target"],
        "constraints": [
            {
                "id": item["id"],
                "text": item["text"],
                "kind": item["kind"],
                "priority": item["priority"],
                "status": item["status"],
                "evidence_docids": item["evidence_docids"][:3],
            }
            for item in state["constraints"]
        ],
        "open_gaps": state["gaps"][:5],
        "searched_queries": state["searched_queries"][-8:],
        "evidence_bank": [
            {
                "docid": item["docid"],
                "source_query": item["source_query"],
                "snippet": item["snippet"][:700],
            }
            for item in state["evidence_bank"][-max_evidence:]
        ],
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def build_system_prompt() -> str:
    return (
        "You are a concise deep research agent for BrowseComp-Plus. "
        "Use only the provided search tool and the evidence in tool results. "
        "Do not answer from memory. You must search before giving a final answer. "
        "Keep track of constraints, evidence, and unresolved gaps. "
        "Call search when any important constraint is still unsupported. "
        "Only give a final answer when the evidence is sufficient and cite key docids. "
        "Final answer must be in Chinese and include: brief evidence, Exact Answer."
    )


def build_state_message(state: Dict[str, Any]) -> Dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Current research state is below. Continue the research. "
            "If important gaps remain, call search with one short specific query. "
            "If the answer is fully supported, provide the final answer.\n\n"
            f"{compact_state_for_prompt(state)}"
        ),
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


def run_search_tool(
    query: str,
    registry: Dict[str, Callable[..., Any]],
    call_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tool_call = make_search_tool_call(call_id, query)
    executed = execute_tool_call(tool_call, registry)
    return tool_call, executed


def evaluate_stop_condition(
    state: Dict[str, Any],
    final_text: str,
    round_id: int,
    max_rounds: int,
    min_evidence_docs: int = 2,
    min_constraint_coverage: float = 0.35,
) -> Dict[str, Any]:
    constraints = state["constraints"]
    supported = [item for item in constraints if item.get("evidence_docids")]
    coverage = len(supported) / max(1, len(constraints))
    high_priority_gaps = [gap for gap in state["gaps"] if gap.get("priority", 1) >= 3]
    evidence_docids = {item["docid"] for item in state["evidence_bank"]}

    reasons = []
    if not final_text.strip():
        reasons.append("empty_final_text")
    if not state["searched_queries"]:
        reasons.append("no_search_performed")
    if len(evidence_docids) < min_evidence_docs:
        reasons.append("too_few_evidence_docs")
    if coverage < min_constraint_coverage and round_id < max_rounds:
        reasons.append("low_constraint_coverage")
    if high_priority_gaps and round_id < max_rounds:
        reasons.append("high_priority_gaps_remain")

    passed = not reasons
    check = {
        "round_id": round_id,
        "passed": passed,
        "reasons": reasons,
        "coverage": round(coverage, 3),
        "supported_constraints": len(supported),
        "total_constraints": len(constraints),
        "evidence_docs": len(evidence_docids),
    }
    state["stop_checks"].append(check)
    return check


def next_gap_query(state: Dict[str, Any]) -> Optional[str]:
    searched = set(state["searched_queries"])
    for gap in sorted(state["gaps"], key=lambda item: item.get("priority", 1), reverse=True):
        query = gap.get("suggested_query") or make_anchor_query(gap.get("text", ""))
        if query and query not in searched:
            return query
    for query in state.get("initial_search_queries", []):
        if query not in searched:
            return query
    return None


def run_research_agent_v2(
    client: Any,
    model: str,
    query: str,
    tool_specs: List[Dict[str, Any]],
    tool_registry: Dict[str, Callable[..., Any]],
    max_rounds: int = 10,
    max_tokens: int = 1024,
    planning_max_tokens: int = 1200,
    initial_search_count: int = 2,
) -> Dict[str, Any]:
    state = initialize_research_state(
        query=query,
        client=client,
        model=model,
        planning_max_tokens=planning_max_tokens,
        use_model_planner=True,
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": query},
    ]
    trajectory: List[Dict[str, Any]] = []

    for idx, search_query in enumerate(state["initial_search_queries"][:initial_search_count], start=1):
        call_id = f"init_search_{idx}"
        tool_call, executed = run_search_tool(search_query, tool_registry, call_id)
        messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(executed["tool_result"], ensure_ascii=False),
            }
        )
        update_state_with_search_results(state, search_query, executed["tool_result"])
        trajectory.append(
            {
                "round_id": 0,
                "phase": "initial_search",
                "tool_calls": [tool_call],
                "tool_results": [executed],
            }
        )

    messages.append(build_state_message(state))

    for round_id in range(1, max_rounds + 1):
        response = client.simple_chat(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            tools=tool_specs,
            tool_choice="auto",
        )
        message = response["choices"][0]["message"]
        raw_content = message.get("content", "")
        tool_calls = message.get("tool_calls") or []

        step: Dict[str, Any] = {
            "round_id": round_id,
            "phase": "agent_loop",
            "assistant_content": raw_content,
            "tool_calls": tool_calls,
            "usage": response.get("usage", {}),
        }
        trajectory.append(step)

        assistant_message = {"role": "assistant", "content": raw_content}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        if tool_calls:
            tool_results = []
            for tool_call in tool_calls:
                executed = execute_tool_call(tool_call, tool_registry)
                tool_results.append(executed)
                arguments = executed["arguments"]
                search_query = str(arguments.get("query", ""))
                update_state_with_search_results(state, search_query, executed["tool_result"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(executed["tool_result"], ensure_ascii=False),
                    }
                )
            step["tool_results"] = tool_results
            messages.append(build_state_message(state))
            continue

        stop_check = evaluate_stop_condition(state, raw_content, round_id, max_rounds)
        step["stop_check"] = stop_check
        if stop_check["passed"]:
            return {
                "query": query,
                "status": "completed",
                "final_output": raw_content,
                "trajectory": trajectory,
                "messages": messages,
                "state": state,
            }

        forced_query = next_gap_query(state)
        if forced_query is None:
            return {
                "query": query,
                "status": "completed_with_unresolved_gaps",
                "final_output": raw_content,
                "trajectory": trajectory,
                "messages": messages,
                "state": state,
            }

        call_id = f"gap_search_{round_id}"
        tool_call, executed = run_search_tool(forced_query, tool_registry, call_id)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Stop check failed: "
                    f"{', '.join(stop_check['reasons'])}. Continue with the most important gap."
                ),
            }
        )
        messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(executed["tool_result"], ensure_ascii=False),
            }
        )
        update_state_with_search_results(state, forced_query, executed["tool_result"])
        step["forced_search"] = {
            "query": forced_query,
            "tool_call": tool_call,
            "tool_result": executed,
        }
        messages.append(build_state_message(state))

    return {
        "query": query,
        "status": "max_rounds_reached",
        "final_output": "",
        "trajectory": trajectory,
        "messages": messages,
        "state": state,
    }
