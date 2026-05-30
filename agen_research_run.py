import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agent.dataset_utils import load_jsonl
from agent.research_agent import run_research_agent
from agent.tools import build_searcher, get_agent_tool_specs_and_registry
from agent.vllm_client import VLLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the research agent on one BrowseComp-Plus query_id.")
    parser.add_argument("query_id", help="query_id in browsecomp_plus_hard50.jsonl")
    parser.add_argument("--dataset", default="browsecomp_plus_hard50.jsonl", help="Path to the question JSONL file.")
    parser.add_argument(
        "--index-path",
        default="indexes/browsecomp_plus_bm25.sqlite",
        help="Path to the BM25 SQLite index.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI-compatible base URL.")
    parser.add_argument("--model", default="qwen_auto", help="Served model name.")
    parser.add_argument("--api-key", default="dummy", help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--top-k", type=int, default=6, help="Number of search results per search call.")
    parser.add_argument("--snippet-max-chars", type=int, default=1600, help="Maximum characters per search snippet.")
    parser.add_argument("--window-chars", type=int, default=1200, help="Characters returned by get_document_window.")
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum loop-agent rounds.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="max_tokens for all model calls.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to runs/research_agent_<query_id>.json.",
    )
    return parser.parse_args()


def find_row_by_query_id(rows: List[Dict[str, Any]], query_id: str) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("query_id", "")) == str(query_id):
            return row
    raise ValueError(f"query_id not found: {query_id}")


def main() -> None:
    args = parse_args()
    dataset_path = PROJECT_ROOT / args.dataset
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "runs" / f"research_agent_{args.query_id}.json"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    rows = load_jsonl(str(dataset_path))
    row = find_row_by_query_id(rows, args.query_id)

    client = VLLMClient(base_url=args.base_url, api_key=args.api_key)
    searcher = build_searcher(index_path=args.index_path)
    tool_specs, tool_registry = get_agent_tool_specs_and_registry(
        searcher=searcher,
        k=args.top_k,
        snippet_max_chars=args.snippet_max_chars,
        window_chars=args.window_chars,
    )

    result = run_research_agent(
        client=client,
        model=args.model,
        query=row["query"],
        tool_specs=tool_specs,
        tool_registry=tool_registry,
        max_rounds=args.max_rounds,
        max_tokens=args.max_tokens,
    )

    output = {
        "query_id": row["query_id"],
        "query": row["query"],
        "gold_answer": row.get("answer", ""),
        "status": result.get("status", ""),
        "final_output": result.get("final_output", ""),
        "messages": result.get("messages", []),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(result.get("final_output", ""))


if __name__ == "__main__":
    main()
