import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from core.agent.tools import build_searcher, get_search_tool_specs_and_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BrowseComp-Plus search tool directly.")
    parser.add_argument("query", nargs="*", help="Search query. If omitted, enter interactive mode.")
    parser.add_argument("--k", type=int, default=8, help="Number of search results to return.")
    parser.add_argument(
        "--snippet-max-chars",
        type=int,
        default=2000,
        help="Maximum characters per returned snippet.",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON results.")
    return parser.parse_args()


def print_results(query: str, results: List[Dict[str, Any]], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    print(f"\nquery: {query}")
    print(f"num_results: {len(results)}")
    for rank, item in enumerate(results, start=1):
        print("\n" + "=" * 80)
        print(f"rank: {rank}")
        print(f"docid: {item.get('docid', '')}")
        print(f"score: {item.get('score', '')}")
        print(f"url: {item.get('url', '')}")
        print("-" * 80)
        print(item.get("snippet", ""))


def run_query(query: str, index_path: str, k: int, snippet_max_chars: int, as_json: bool) -> None:
    searcher = build_searcher(index_path=index_path)
    _, registry = get_search_tool_specs_and_registry(
        searcher=searcher,
        k=k,
        snippet_max_chars=snippet_max_chars,
    )
    results = registry["search"](query=query)
    print_results(query, results, as_json=as_json)


def main() -> None:
    args = parse_args()
    index_path = "indexes/browsecomp_plus_bm25.sqlite"
    query = " ".join(args.query).strip()

    if query:
        run_query(query, index_path, args.k, args.snippet_max_chars, args.json)
        return

    print("Enter a search query. Press Ctrl+C or submit an empty line to exit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            return
        run_query(query, index_path, args.k, args.snippet_max_chars, args.json)


if __name__ == "__main__":
    main()
