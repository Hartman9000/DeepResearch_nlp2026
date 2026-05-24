import re
from typing import Any, Callable, Dict, List, Tuple

from .browsecomp_searcher import BrowseCompBM25Searcher, snippetize


SNIPPET_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "with",
}


def build_searcher(index_path: str) -> BrowseCompBM25Searcher:
    return BrowseCompBM25Searcher(index_path=index_path)


def _query_terms(query: str) -> List[str]:
    quoted_phrases = [
        phrase.strip().lower()
        for phrase in re.findall(r"['\"]([^'\"]{3,100})['\"]", query)
        if phrase.strip()
    ]
    tokens = re.findall(
        r"[A-Za-z][A-Za-z0-9'\-]*|[$]?\d[\d,]*(?:\.\d+)?(?:s|%)?",
        query.lower(),
    )

    terms = []
    seen = set()
    for term in quoted_phrases + tokens:
        term = term.strip("'-. ")
        if len(term) <= 2 and not term.isdigit():
            continue
        if term in SNIPPET_STOPWORDS:
            continue
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def snippetize_around_query(text: str, query: str, max_chars: int = 1200) -> str:
    if not max_chars or max_chars <= 0 or len(text) <= max_chars:
        return text

    lowered = text.lower()
    positions = []
    for term in _query_terms(query):
        start = 0
        while True:
            idx = lowered.find(term, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + max(1, len(term))
            if len(positions) >= 200:
                break
        if len(positions) >= 200:
            break

    if not positions:
        return snippetize(text, max_chars)

    window_radius = max_chars // 2
    best_position = min(
        positions,
        key=lambda pos: (
            -sum(abs(pos - other) <= window_radius for other in positions),
            pos,
        ),
    )
    start = max(0, best_position - window_radius)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet.rstrip() + "..."
    return snippet


def retrieve_once(
    searcher: BrowseCompBM25Searcher,
    query: str,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    docs = searcher.search(query, k=k)
    return [
        {
            "docid": doc["docid"],
            "score": doc["score"],
            "snippet": snippetize_around_query(doc["text"], query, snippet_max_chars),
            "url": doc.get("url", ""),
        }
        for doc in docs
    ]


def format_rag_context(results: List[Dict[str, Any]]) -> str:
    blocks = []
    for rank, item in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Document {rank}]",
                    f"docid: {item['docid']}",
                    f"score: {item['score']}",
                    f"url: {item.get('url', '')}",
                    item["snippet"],
                ]
            )
        )
    return "\n\n".join(blocks)


def get_search_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str) -> List[Dict[str, Any]]:
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    f"Search the BrowseComp-Plus BM25 index and return top-{k} results "
                    "with docid, score, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    return tools, {"search": search}


def get_agent_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str) -> List[Dict[str, Any]]:
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

    def get_document(docid: str) -> Dict[str, Any]:
        doc = searcher.get_document(docid)
        if doc is None:
            return {"docid": docid, "error": "document not found"}
        return doc

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    f"Search the BrowseComp-Plus BM25 index and return top-{k} results "
                    "with docid, score, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_document",
                "description": "Retrieve a full document by its docid.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id"},
                    },
                    "required": ["docid"],
                },
            },
        },
    ]
    return tools, {"search": search, "get_document": get_document}
