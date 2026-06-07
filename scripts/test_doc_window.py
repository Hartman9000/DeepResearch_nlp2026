import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agent.tools import build_searcher, get_document_window_tool_specs_and_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a keyword-centered window from one BrowseComp-Plus document."
    )
    parser.add_argument("docid", help="Document id to inspect.")
    parser.add_argument("keyword", nargs="+", help="Keyword or phrase to locate.")
    parser.add_argument(
        "--index-path",
        default="indexes/browsecomp_plus_bm25.sqlite",
        help="Path to the SQLite BM25 index.",
    )
    parser.add_argument(
        "--window-chars",
        type=int,
        default=600,
        help="Maximum characters in the returned window.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use case-sensitive keyword matching.",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=3,
        help="Maximum number of keyword matches to return.",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON result.")
    return parser.parse_args()


def print_result(result: List[Dict[str, Any]], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not result:
        return
    item = result[0]
    print(f"docid: {item.get('docid', '')}")
    print(f"keyword: {item.get('keyword', '')}")
    print(f"found: {item.get('found', False)}")
    if item.get("error"):
        print(f"error: {item['error']}")
        return
    if not item.get("found"):
        return
    print("\n")
    print(item.get("snippet", ""))
    print("\n")

def main() -> None:
    args = parse_args()
    keyword = " ".join(args.keyword).strip()
    searcher = build_searcher(index_path=args.index_path)
    _, registry = get_document_window_tool_specs_and_registry(
        searcher=searcher,
        window_chars=args.window_chars,
        case_sensitive=args.case_sensitive,
        max_matches=args.max_matches,
    )
    result = registry["get_document_window"](docid=args.docid, keyword=keyword)
    print_result(result, as_json=args.json)


if __name__ == "__main__":
    main()
