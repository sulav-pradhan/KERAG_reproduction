import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from src import config

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = (
    "KERAG-Head2Tail-Reproduction/1.0"
    " (academic-research@wright.edu)"
)
ABSTENTION_PHRASES = (
    "i don't know",
    "i do not know",
    "not known",
    "unknown",
    "no information",
    "not provided",
)


def normalize_entity_title(entity_or_uri: str) -> str:
  """Converts a DBpedia URI/slug like 'Hasta_el_fin_del_mundo' into 'Hasta el fin del mundo'."""
  s = entity_or_uri.strip()
  if s.startswith("<") and s.endswith(">"):
    s = s[1:-1]
  title = s.rstrip("/").split("/")[-1]
  title = title.replace("_", " ")
  return re.sub(r"\s+", " ", title).strip()


def is_abstention(answer: str | None) -> bool:
  """Returns True if a model answer represents an abstention / unknown response."""
  if not answer:
    return True
  cleaned = " ".join(answer.strip().lower().split()).rstrip(".!?")
  if not cleaned:
    return True
  return any(phrase in cleaned for phrase in ABSTENTION_PHRASES)


def should_escalate(
    num_filtered_triples: int, tier1_answer: str | None
) -> bool:
  """Escalate to Tier-2 Wikipedia fallback when 1-hop KG yields 0 triples or Tier-1 abstains."""
  return num_filtered_triples == 0 or is_abstention(tier1_answer)


def load_wiki_cache(cache_path: Path = config.WIKI_CACHE_PATH) -> dict[str, str]:
  """Loads cached {normalized_title: plain_text} entries from JSONL."""
  cache: dict[str, str] = {}
  if not cache_path.exists():
    return cache
  with open(cache_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      if isinstance(rec, dict) and "title" in rec and "text" in rec:
        cache[rec["title"]] = rec["text"]
  return cache


def append_wiki_cache(
    title: str, text: str, cache_path: Path = config.WIKI_CACHE_PATH
) -> None:
  """Appends a single Wikipedia fetch result to the persistent JSONL cache."""
  cache_path.parent.mkdir(parents=True, exist_ok=True)
  with open(cache_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n")


def _request_with_backoff(
    session: requests.Session,
    params: dict[str, Any],
    user_agent: str = DEFAULT_USER_AGENT,
    max_retries: int = 4,
    timeout: float = 15.0,
) -> dict[str, Any] | None:
  """Performs a polite MediaWiki API GET request with 429/503 backoff."""
  headers = {"User-Agent": user_agent}
  for attempt in range(max_retries):
    try:
      resp = session.get(
          WIKI_API_URL, params=params, headers=headers, timeout=timeout
      )
      if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
          err_code = data["error"].get("code", "")
          if err_code == "maxlag" and attempt < max_retries - 1:
            time.sleep(2.0 * (attempt + 1))
            continue
          return None
        return data if isinstance(data, dict) else None
      if resp.status_code in (429, 503):
        retry_after = resp.headers.get("Retry-After")
        wait_sec = (
            float(retry_after)
            if retry_after and retry_after.isdigit()
            else 2.0 * (2**attempt)
        )
        time.sleep(min(wait_sec, 30.0))
        continue
      return None
    except requests.RequestException:
      if attempt < max_retries - 1:
        time.sleep(1.5 * (attempt + 1))
        continue
      return None
  return None


def _search_best_title(
    session: requests.Session, query_title: str, user_agent: str
) -> str | None:
  """Uses MediaWiki opensearch as a fallback title resolver when direct lookup misses."""
  data = _request_with_backoff(
      session,
      params={
          "action": "opensearch",
          "search": query_title,
          "limit": 3,
          "namespace": 0,
          "format": "json",
      },
      user_agent=user_agent,
  )
  if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
    for candidate in data[1]:
      if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
  return None


def fetch_wiki_article(
    entity_or_uri: str,
    cache: dict[str, str],
    cache_path: Path = config.WIKI_CACHE_PATH,
    session: requests.Session | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    min_delay_sec: float = 0.35,
) -> str:
  """Fetches plain-text Wikipedia extract for an entity, using and updating cache."""
  norm_key = normalize_entity_title(entity_or_uri).lower()
  if not norm_key:
    return ""
  if norm_key in cache:
    return cache[norm_key]

  own_session = session is None
  active_session = session or requests.Session()
  try:
    query_title = normalize_entity_title(entity_or_uri)
    extract_text = ""

    for attempt_title in (query_title,):
      data = _request_with_backoff(
          active_session,
          params={
              "action": "query",
              "prop": "extracts",
              "explaintext": 1,
              "redirects": 1,
              "maxlag": 5,
              "titles": attempt_title,
              "format": "json",
              "formatversion": 2,
          },
          user_agent=user_agent,
      )
      pages = data.get("query", {}).get("pages", []) if data else []
      if pages and not pages[0].get("missing", False):
        extract_text = (pages[0].get("extract") or "").strip()

    if not extract_text:
      resolved_title = _search_best_title(
          active_session, query_title, user_agent
      )
      if resolved_title and resolved_title.lower() != query_title.lower():
        data = _request_with_backoff(
            active_session,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
                "maxlag": 5,
                "titles": resolved_title,
                "format": "json",
                "formatversion": 2,
            },
            user_agent=user_agent,
        )
        pages = data.get("query", {}).get("pages", []) if data else []
        if pages and not pages[0].get("missing", False):
          extract_text = (pages[0].get("extract") or "").strip()

    cache[norm_key] = extract_text
    append_wiki_cache(norm_key, extract_text, cache_path=cache_path)
    if min_delay_sec > 0:
      time.sleep(min_delay_sec)
    return extract_text
  finally:
    if own_session:
      active_session.close()


def chunk_text(
    text: str,
    chunk_words: int = 150,
    overlap_words: int = 30,
    max_chunks: int = 40,
) -> list[str]:
  """Splits plain text into overlapping word-window chunks capped at max_chunks."""
  if not text or not text.strip():
    return []
  words = text.split()
  if len(words) <= chunk_words:
    return [" ".join(words)]

  step = max(1, chunk_words - overlap_words)
  chunks: list[str] = []
  for start in range(0, len(words), step):
    window = words[start : start + chunk_words]
    if not window:
      break
    chunks.append(" ".join(window))
    if len(chunks) >= max_chunks:
      break
    if start + chunk_words >= len(words):
      break
  return chunks


def rank_chunks_dpr(
    question: str, chunks: list[str], dpr_retriever: Any, top_k: int = 3
) -> list[tuple[str, float]]:
  """Ranks chunks using a DPRRetriever instance and returns top_k (chunk, score) pairs."""
  if not chunks:
    return []
  k_sel = min(top_k, len(chunks))
  ranked = dpr_retriever.retrieve_top_k(question, chunks, k=k_sel)
  return [(chunks[idx], float(score)) for idx, score in ranked]


def rank_chunks_lexical(
    question: str, chunks: list[str], top_k: int = 3
) -> list[tuple[str, float]]:
  """CPU-only token-overlap fallback ranker for local testing without DPR weights."""
  if not chunks:
    return []
  q_tokens = set(re.findall(r"\w+", question.lower()))
  if not q_tokens:
    return [(c, 0.0) for c in chunks[:top_k]]

  scored: list[tuple[str, float]] = []
  for c in chunks:
    c_tokens = re.findall(r"\w+", c.lower())
    if not c_tokens:
      scored.append((c, 0.0))
      continue
    overlap = sum(1 for tok in c_tokens if tok in q_tokens)
    score = overlap / (len(c_tokens) ** 0.5)
    scored.append((c, score))

  scored.sort(key=lambda item: item[1], reverse=True)
  return scored[: min(top_k, len(scored))]