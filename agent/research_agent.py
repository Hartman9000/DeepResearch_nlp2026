import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple


VALID_PRIORITIES = {"critical", "strong", "weak"}
VALID_STATUSES = {"unknown", "supported", "contradicted"}

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
    "in",
    "inclusive",
    "is",
    "it",
    "its",
    "of",
    "on",
    "one",
    "or",
    "she",
    "some",
    "that",
    "the",
    "their",
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
}


PARSE_PROMPT = """You are parse_agent for BrowseComp-Plus style questions.
Return strict JSON only. Do not answer the question.

Your job:
1. Convert the question into short atomic constraints.
2. Generate anchor BM25 queries likely to retrieve answer-related snippets.

Constraint rules:
- Each constraint must be one short checkable fact.
- Use priority "critical" for facts required to identify the answer.
- Use priority "strong" for facts that strongly distinguish candidates.
- Use priority "weak" for helpful but nonessential clues.
- Every constraint must start with status "unknown".

Anchor query rules:
- Never search the full question.
- Do not write natural-language questions.
- Use high-information tokens only.
- Generate 1-3 anchor queries only.
- Each anchor query should be medium-long, usually 6-12 high-information terms.
- Prefer rare phrases, names, places, exact years, prices, titles, and relationship anchors.
- Do not include numeric ranges or decade expressions such as "1980s", "1920s", or "1900-1910".
- Prefer distinctive clue clusters over broad category words.

Example BrowseComp-Plus style question, for query design only:
"A privately printed travel book mentions a lighthouse keeper's daughter who translated a mayor's diaries. The same author later wrote a biography whose preface thanks a ceramicist and a harbor archivist. What is the dedication line in that biography?"
Good anchor_queries:
[
  "lighthouse keeper daughter translated mayor diaries",
  "biography preface thanks ceramicist harbor archivist",
  "privately printed travel book mayor diaries biography"
]

Return this schema:
{
  "target": {"answer_type": "...", "description": "..."},
  "constraints": [
    {"id": "c1", "text": "...", "priority": "critical", "status": "unknown"}
  ],
  "anchor_queries": ["rare terms query", "..."]
}
"""


EXTRACT_EVIDENCE_PROMPT = """You are extract_evidence_agent.
Return strict JSON only. Use only the provided snippets.

Your job:
1. Select 1-3 docids most relevant to the original question.
2. Update constraint status when the snippet directly supports or contradicts it.
3. Record candidate answers only when a snippet directly suggests one.
4. Use analysis_log as prior reasoning context, but ground every evidence update in snippets.

Relevance should be judged from several angles:
- Direct answer: the snippet contains, or is very likely to contain after a document-window check,
  the target answer.
- Semantic/topic match: the snippet is semantically about the same entity, work, event, place,
  relationship, or clue in the question. Do not rely on keyword overlap alone.
- Bridge value: the snippet connects key entities, such as a work to its author, an author to
  another work, or a source page to a table of contents.
- No redundancy: prefer compact snippets with useful facts. Avoid snippets that contain lots of
  unrelated material or only match generic words, dates, page numbers, or common phrases.
- Relation to existing evidence: a new snippet can be important if it connects to an already
  evident docid or presumed entity, even if it is not the final answer.

Rules:
- Do not use outside knowledge.
- Prefer direct quotes or faithful short summaries from snippets.
- Mark a constraint supported only when evidence is explicit.
- Leave ambiguous constraints unknown.
- A candidate answer must have direct evidence_docids.

Return this schema:
{
  "selected_snippets": [
    {
      "docid": "123",
      "why": "why this snippet matters",
    }
  ],
  "constraint_updates": [
    {
      "id": "c1",
      "status": "supported",
      "evidence_docids": ["123"],
      "rationale": "brief reason"
    }
  ],
  "candidate_answers": [
    {
      "answer": "...",
      "confidence": "low|medium|high",
      "evidence_docids": ["123"],
      "rationale": "brief reason"
    }
  ],
  "analysis": "brief visible analysis of what the new evidence changes"
}
"""


LOOP_PROMPT = """You are loop_agent for a simple deep research agent.

You will receive the current work_message: original question, constraints, evident snippets,
candidate answers, tool history, and previous visible analyses.

Your response content is your current visible analysis. Keep it concise and useful.
Decide what to investigate next from the current state.

Available tools:
- search(query): discover candidate documents, bridge entities, names, titles, dates, or source pages.
- get_document_window(docid, keyword): inspect a known document around one keyword. The keyword must be
  exactly one word with no spaces. Use it to verify
  precise clues such as acknowledgements, chapter headings, page-like references, names, dates, prices,
  and distinctive phrases inside a document.

Tool-use rules:
- You may call one or more tools when more investigation is useful.
- Never search the full original question.
- Do not write natural-language search questions.
- Use compact high-information search terms.
- Prefer discovered bridge entities, titles, names, years, distinctive phrases, and relation anchors.
- Use get_document_window when you already have a promising docid and need local evidence inside it.
- For get_document_window, pass a single distinctive word only, for example "acknowledgements",
  "chapter", "spear", "barrel", or a surname. Do not pass phrases like "first chapter".
- Avoid repeating equivalent tool calls from tool_history.
- If you use <think>...</think>, you must write a short visible analysis after </think> before any tool call.
- Your post-think visible content must not be empty. Use the format:
  <think>private reasoning</think>
  Analysis: concise reason for the next tool call or final answer.

Final-answer rules:
- If a candidate answer has direct evidence, all critical constraints are supported, most strong constraints
  are supported, and there is no contradiction, stop calling tools and answer in English.
- Final answer must include brief evidence with docids, an `Exact Answer:` line, and a `Confidence:` line.
"""


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction for small local models that add prose."""
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


def append_agent_exchange(
    messages: List[Dict[str, Any]],
    system_prompt: str,
    user_content: str,
    assistant_content: str,
) -> None:
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})


def call_json_agent(
    client: Any,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
) -> Tuple[Dict[str, Any], str]:
    response = client.simple_chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    raw = response["choices"][0]["message"].get("content", "")
    parsed = extract_json_object(raw)
    if parsed is None:
        raise ValueError(f"Model did not return valid JSON:\n{raw}")
    return parsed, raw


def normalize_priority(value: Any, default: str = "strong") -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in VALID_PRIORITIES:
            return lowered
        if lowered in {"high", "must", "required"}:
            return "critical"
        if lowered in {"medium", "normal"}:
            return "strong"
        if lowered in {"low", "minor"}:
            return "weak"
    if isinstance(value, int):
        if value >= 3:
            return "critical"
        if value == 2:
            return "strong"
        return "weak"
    return default


def normalize_status(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in VALID_STATUSES:
        return value.strip().lower()
    return "unknown"


def keyword_tokens(text: str) -> List[str]:
    tokens = re.findall(
        r"[A-Za-z][A-Za-z0-9'\-]*|[$]?\d[\d,]*(?:\.\d+)?(?:s|%)?",
        text.lower(),
    )
    cleaned: List[str] = []
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
    terms = keyword_tokens(text)
    return " ".join(terms[:max_terms]).strip()


def normalize_query(query: Any, original_query: str = "") -> str:
    text = " ".join(str(query).split())
    text = text.strip(" \t\r\n\"'")
    if not text:
        return ""
    if original_query and text.lower() == " ".join(original_query.lower().split()):
        return ""

    tokens = text.split()
    if len(tokens) > 12:
        shortened = make_anchor_query(text, max_terms=12)
        return shortened or " ".join(tokens[:12])
    return text


def remove_range_numbers(text: str) -> str:
    text = re.sub(r"\b\d{3,4}\s*(?:-|–|—|to)\s*\d{2,4}\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{3,4}'?s\b", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def normalize_constraints(raw_constraints: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_constraints, list) or not raw_constraints:
        raise ValueError("parse_agent output must include a non-empty constraints list.")

    constraints: List[Dict[str, Any]] = []
    seen_ids = set()
    for idx, item in enumerate(raw_constraints[:14], start=1):
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        constraint_id = str(item.get("id") or f"c{idx}").strip() or f"c{idx}"
        if constraint_id in seen_ids:
            constraint_id = f"c{idx}"
        seen_ids.add(constraint_id)
        constraints.append(
            {
                "id": constraint_id,
                "text": text,
                "priority": normalize_priority(item.get("priority"), "critical" if idx <= 2 else "strong"),
                "status": "unknown",
                "evidence_docids": [],
                "rationale": "",
            }
        )

    if not constraints:
        raise ValueError("parse_agent produced no valid constraints.")
    return constraints


def normalize_anchor_queries(raw_queries: Any, original_query: str) -> List[str]:
    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError("parse_agent output must include a non-empty anchor_queries list.")

    queries: List[str] = []
    seen = set()
    for item in raw_queries[:3]:
        query = normalize_query(remove_range_numbers(str(item)), original_query=original_query)
        query = remove_range_numbers(query)
        if query and query.lower() not in seen:
            seen.add(query.lower())
            queries.append(query)

    if not queries:
        raise ValueError("parse_agent produced no valid anchor queries.")
    return queries


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


def make_search_tool_call(call_id: str, query: str) -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "search",
            "arguments": json.dumps({"query": query}, ensure_ascii=False),
        },
    }


def execute_tool_call(tool_call: Dict[str, Any], registry: Dict[str, Callable[..., Any]]) -> Dict[str, Any]:
    function = tool_call.get("function", {})
    name = function.get("name", "")
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if name not in registry:
        raise ValueError(f"Unknown tool: {name}")
    return {
        "tool_name": name,
        "arguments": arguments,
        "tool_result": registry[name](**arguments),
    }


def merge_search_results(
    snippet_bank: Dict[str, Dict[str, Any]],
    search_query: str,
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    touched: List[Dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        if isinstance(result, dict) and result.get("found") is False:
            continue
        docid = str(result.get("docid", "")).strip()
        if not docid:
            continue
        snippet = str(result.get("snippet", "")).strip()
        entry = snippet_bank.get(docid)
        if entry is None:
            entry = {
                "docid": docid,
                "url": result.get("url", ""),
                "best_score": result.get("score", 0.0),
                "retrieval_count": 0,
                "source_queries": [],
                "snippets": [],
                "best_rank": rank,
            }
            snippet_bank[docid] = entry
        entry["retrieval_count"] += 1
        if search_query not in entry["source_queries"]:
            entry["source_queries"].append(search_query)
        if snippet and snippet not in entry["snippets"]:
            entry["snippets"].append(snippet)
        try:
            if float(result.get("score", 0.0)) > float(entry.get("best_score", 0.0)):
                entry["best_score"] = result.get("score", 0.0)
        except (TypeError, ValueError):
            pass
        entry["best_rank"] = min(int(entry.get("best_rank", rank)), rank)
        touched.append(entry)
    return touched


def compact_snippet(entry: Dict[str, Any]) -> Dict[str, Any]:
    snippets = entry.get("snippets", [])
    text = "\n---\n".join(str(item) for item in snippets)
    return {
        "docid": entry["docid"],
        "url": entry.get("url", ""),
        "best_score": entry.get("best_score", 0.0),
        "retrieval_count": entry.get("retrieval_count", 0),
        "source_queries": entry.get("source_queries", []),
        "snippet": text,
    }


def run_search(
    query: str,
    tool_registry: Dict[str, Callable[..., Any]],
    call_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tool_call = make_search_tool_call(call_id, query)
    executed = execute_tool_call(tool_call, tool_registry)
    return tool_call, executed


def summarize_tool_result(executed: Dict[str, Any]) -> Dict[str, Any]:
    name = executed.get("tool_name", "")
    result = executed.get("tool_result")
    if isinstance(result, list):
        summary = {
            "num_results": len(result),
            "docids": [str(item.get("docid", "")) for item in result[:8] if isinstance(item, dict)],
        }
        if any(isinstance(item, dict) and "found" in item for item in result):
            summary["found"] = any(isinstance(item, dict) and item.get("found") for item in result)
            summary["keywords"] = [
                str(item.get("keyword", ""))
                for item in result[:8]
                if isinstance(item, dict) and item.get("keyword")
            ]
        return summary
    return {"type": type(result).__name__}


def merge_tool_result(
    state: Dict[str, Any],
    snippet_bank: Dict[str, Dict[str, Any]],
    executed: Dict[str, Any],
) -> List[Dict[str, Any]]:
    name = executed.get("tool_name", "")
    arguments = executed.get("arguments", {})
    result = executed.get("tool_result")
    state["tool_history"].append(
        {
            "tool_name": name,
            "arguments": arguments,
            "summary": summarize_tool_result(executed),
        }
    )

    if isinstance(result, list):
        query = normalize_query(
            arguments.get("query") or arguments.get("q") or arguments.get("keyword") or name,
            state["original_query"],
        )
        if query:
            state["searched_queries"].append(query)
        return merge_search_results(snippet_bank, query, result)

    return []


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
