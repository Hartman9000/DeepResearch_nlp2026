import json
from typing import Any, Callable, Dict, List, Tuple

from .normalization import normalize_query


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
