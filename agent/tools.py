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
    term_positions: Dict[str, List[int]] = {}
    for term in _query_terms(query):
        start = 0
        positions = []
        while True:
            idx = lowered.find(term, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + max(1, len(term))
            if len(positions) >= 200:
                break
        if positions:
            term_positions[term] = positions

    if not term_positions:
        return snippetize(text, max_chars)

    window_radius = max_chars // 2
    weighted_positions = []
    for term, positions in term_positions.items():
        frequency = max(1, len(positions))
        weight = max(1.0, len(term) / (frequency ** 0.5))
        for pos in positions:
            weighted_positions.append((pos, term, weight))

    def window_score(candidate_pos: int) -> Tuple[float, int]:
        score = 0.0
        rare_hits = 0
        for pos, term, weight in weighted_positions:
            if abs(candidate_pos - pos) <= window_radius:
                score += weight
                if len(term_positions.get(term, [])) <= 3:
                    rare_hits += 1
        return score, rare_hits

    best_position = max(
        weighted_positions,
        key=lambda item: (window_score(item[0]), -item[0]),
    )[0]
    start = max(0, best_position - window_radius)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet.rstrip() + "..."
    return snippet


def get_document_keyword_window(
    searcher: BrowseCompBM25Searcher,
    docid: str,
    keyword: str,
    window_chars: int = 600,
    case_sensitive: bool = False,
    max_matches: int = 3,
) -> List[Dict[str, Any]]:
    keyword = str(keyword).strip()
    if not keyword:
        return [{
            "docid": str(docid),
            "keyword": keyword,
            "found": False,
            "error": "keyword is empty",
            "snippet": "",
        }]
    if len(keyword.split()) != 1:
        return [{
            "docid": str(docid),
            "keyword": keyword,
            "found": False,
            "error": "keyword must be a single word with no spaces",
            "snippet": "",
        }]

    doc = searcher.get_document(docid)
    if doc is None:
        return [{
            "docid": str(docid),
            "keyword": keyword,
            "found": False,
            "error": "document not found",
            "snippet": "",
        }]

    text = doc["text"]
    haystack = text if case_sensitive else text.lower()
    needle = keyword if case_sensitive else keyword.lower()
    max_matches = max(1, int(max_matches))
    window_chars = max(1, int(window_chars))

    match_starts = []
    search_from = 0
    while len(match_starts) < max_matches:
        match_start = haystack.find(needle, search_from)
        if match_start == -1:
            break
        match_starts.append(match_start)
        search_from = match_start + max(1, len(needle))

    if not match_starts:
        return [{
            "docid": doc["docid"],
            "keyword": keyword,
            "found": False,
            "snippet": "",
        }]

    matches = []
    for match_index, match_start in enumerate(match_starts, start=1):
        match_end = match_start + len(keyword)
        center = match_start + max(1, len(keyword)) // 2
        window_start = max(0, center - window_chars // 2)
        window_end = min(len(text), window_start + window_chars)
        window_start = max(0, window_end - window_chars)
        window = text[window_start:window_end].strip()
        if window_start > 0:
            window = "..." + window
        if window_end < len(text):
            window = window.rstrip() + "..."
        matches.append(window)

    return [{
        "docid": doc["docid"],
        "keyword": keyword,
        "found": True,
        "snippet": "\n\n".join(matches),
    }]


# def retrieve_once(
#     searcher: BrowseCompBM25Searcher,
#     query: str,
#     k: int = 5,
#     snippet_max_chars: int = 1500,
# ) -> List[Dict[str, Any]]:
#     docs = searcher.search(query, k=k)
#     return [
#         {
#             "docid": doc["docid"],
#             "score": doc["score"],
#             "snippet": snippetize_around_query(doc["text"], query, snippet_max_chars),
#             "url": doc.get("url", ""),
#         }
#         for doc in docs
#     ]

def retrieve_once(
    searcher: BrowseCompBM25Searcher,
    query: str,
    k: int = 6,
    snippet_max_chars: int = 1600,
) -> List[Dict[str, Any]]:
    docs = searcher.search(query, k=k)
    return [
        {
            "docid": doc["docid"],
            "score": doc["score"],
            "snippet": snippetize(doc["text"], snippet_max_chars),
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


def get_document_window_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    window_chars: int = 1200,
    case_sensitive: bool = False,
    max_matches: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def get_document_window(docid: str, keyword: str) -> List[Dict[str, Any]]:
        return get_document_keyword_window(
            searcher=searcher,
            docid=docid,
            keyword=keyword,
            window_chars=window_chars,
            case_sensitive=case_sensitive,
            max_matches=max_matches,
        )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_document_window",
                "description": (
                    "Retrieve short windows around up to the first three occurrences "
                    "of one single-word keyword inside one BrowseComp-Plus document."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id"},
                        "keyword": {
                            "type": "string",
                            "description": "Single word to locate in the document; spaces are not allowed",
                        },
                    },
                    "required": ["docid", "keyword"],
                },
            },
        }
    ]
    return tools, {"get_document_window": get_document_window}


def get_agent_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
    window_chars: int = 1200,
    case_sensitive: bool = False,
    max_matches: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str) -> List[Dict[str, Any]]:
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

    def get_document_window(docid: str, keyword: str) -> List[Dict[str, Any]]:
        return get_document_keyword_window(
            searcher=searcher,
            docid=docid,
            keyword=keyword,
            window_chars=window_chars,
            case_sensitive=case_sensitive,
            max_matches=max_matches,
        )

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
                "name": "get_document_window",
                "description": (
                    "Retrieve short windows around up to the first three occurrences "
                    "of one single-word keyword inside one BrowseComp-Plus document."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id"},
                        "keyword": {
                            "type": "string",
                            "description": "Single word to locate in the document; spaces are not allowed",
                        },
                    },
                    "required": ["docid", "keyword"],
                },
            },
        },
    ]
    return tools, {"search": search, "get_document_window": get_document_window}
