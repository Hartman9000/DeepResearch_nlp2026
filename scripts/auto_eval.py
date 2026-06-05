import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [script_path.parent, *script_path.parents]

    for candidate in candidates:
        if (candidate / "agent").is_dir() and (candidate / "browsecomp_plus_hard50.jsonl").exists():
            return candidate

    for candidate in candidates:
        if (candidate / "agent").is_dir():
            return candidate

    return Path.cwd()


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.dataset_utils import load_jsonl
from agent.eval import run_evaluation
from agent.tools import build_searcher, get_agent_tool_specs_and_registry
from agent.vllm_client import VLLMClient
from open_track.research_agent import run_research_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current research agent on BrowseComp-Plus hard50 and evaluate the 50 results."
    )
    parser.add_argument("--dataset", default="browsecomp_plus_hard50.jsonl", help="BrowseComp-Plus hard50 JSONL path.")
    parser.add_argument("--index-path", default="indexes/browsecomp_plus_bm25.sqlite", help="BM25 SQLite index path.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI-compatible base URL.")
    parser.add_argument("--model", default="qwen_auto", help="Model name for the research agent and default judge.")
    parser.add_argument("--eval-model", default=None, help="Judge model name. Defaults to --model.")
    parser.add_argument("--api-key", default="dummy", help="API key for the vLLM endpoint.")
    parser.add_argument("--top-k", type=int, default=6, help="Search results per search call.")
    parser.add_argument("--snippet-max-chars", type=int, default=1600, help="Maximum characters per search snippet.")
    parser.add_argument("--window-chars", type=int, default=1200, help="Characters returned by get_document_window.")
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum research-agent loop rounds.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="max_tokens for research-agent model calls.")
    parser.add_argument("--eval-max-tokens", type=int, default=4096, help="max_tokens for judge calls.")
    parser.add_argument("--output-dir", default="runs", help="Directory for submission, eval, and summary files.")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def build_error_record(row: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
    return {
        "query_id": row.get("query_id", ""),
        "query": row.get("query", ""),
        "gold_answer": row.get("answer", ""),
        "status": "error",
        "predicted_answer": "",
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "messages": [
            {"role": "system", "content": "Research-agent run failed."},
            {"role": "user", "content": row.get("query", "")},
            {"role": "assistant", "content": f"ERROR: {repr(exc)}"},
        ],
    }


def run_predictions(args: argparse.Namespace, rows: List[Dict[str, Any]], submission_path: Path) -> None:
    client = VLLMClient(base_url=args.base_url, api_key=args.api_key)
    searcher = build_searcher(index_path=str(resolve_path(args.index_path)))
    tool_specs, tool_registry = get_agent_tool_specs_and_registry(
        searcher=searcher,
        k=args.top_k,
        snippet_max_chars=args.snippet_max_chars,
        window_chars=args.window_chars,
    )

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    with submission_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows, start=1):
            query_id = str(row.get("query_id", ""))
            print(f"[predict {idx:02d}/{len(rows):02d}] query_id={query_id}")

            try:
                result = run_research_agent(
                    client=client,
                    model=args.model,
                    query=row["query"],
                    tool_specs=tool_specs,
                    tool_registry=tool_registry,
                    max_rounds=args.max_rounds,
                    max_tokens=args.max_tokens,
                )
                record = {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "gold_answer": row.get("answer", ""),
                    "status": result.get("status", ""),
                    "predicted_answer": result.get("final_output", ""),
                    "messages": result.get("messages", []),
                }
            except Exception as exc:
                print(f"  ERROR: {repr(exc)}")
                record = build_error_record(row, exc)

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            preview = str(record.get("predicted_answer", "")).replace("\n", " ")[:180]
            print(f"  status={record.get('status')} pred={preview}")


def save_summary(summary_path: Path, summary: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 50)
    print("Research Agent hard50 Evaluation")
    print("=" * 50)
    print(f"total_queries: {summary['total_queries']}")
    print(f"correct:       {summary['correct']}")
    print(f"incorrect:     {summary['incorrect']}")
    print(f"accuracy:      {summary['accuracy']:.2%}")
    print(f"avg_tools:     {summary['avg_tool_calls_per_query']}")
    print(f"avg_docs:      {summary['avg_retrieved_docs_per_query']}")
    print(f"eval_model:    {summary['eval_model']}")


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    output_dir = resolve_path(args.output_dir)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    submission_path = output_dir / f"submission_{timestamp}.jsonl"
    eval_path = output_dir / f"eval_{timestamp}.jsonl"
    summary_path = output_dir / f"summary_{timestamp}.json"

    rows = load_jsonl(dataset_path, limit=50)
    if len(rows) != 50:
        raise ValueError(f"Expected 50 rows from {dataset_path}, got {len(rows)}.")

    run_predictions(args, rows, submission_path)
    print(f"\nSubmission saved to: {submission_path}")

    summary, _details = run_evaluation(
        submission_path=str(submission_path),
        dataset_path=str(dataset_path),
        model_name=args.eval_model or args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        output_path=str(eval_path),
        max_tokens=args.eval_max_tokens,
        verbose=True,
    )
    save_summary(summary_path, summary)
    print_summary(summary)
    print(f"\nEvaluation saved to: {eval_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
