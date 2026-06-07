import re
from typing import Any, Dict, List


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
