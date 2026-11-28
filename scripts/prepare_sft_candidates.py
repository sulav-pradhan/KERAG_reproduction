import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_utils import extract_entity_name, load_htt_splits, normalize_text
from src.eval_utils import has_answer_in_passages
from src.wiki_fallback import (
    DEFAULT_USER_AGENT,
    chunk_text,
    fetch_wiki_article,
    load_wiki_cache,
    rank_chunks_lexical,
)

DBPEDIA_SPARQL_URL = "https://dbpedia.org/sparql"
SFT_KG_CACHE_PATH = config.DBPEDIA_CACHE_PATH.parent / "sft_kg_cache.jsonl"


def load_blocklist(
    blocklist_path: Path = config.TEST_BLOCKLIST_PATH,
) -> tuple[set[str], set[str]]:
  with open(blocklist_path, "r", encoding="utf-8") as f:
    data = json.load(f)
  blocked_q = {normalize_text(q) for q in data.get("questions", [])}
  blocked_e = {normalize_text(e) for e in data.get("entities", [])}
  return blocked_q, blocked_e


def _record_kg_entry(
    kg_cache: dict[str, list[tuple[str, str, str]]],
    raw_entry: Any,
) -> None:
  if not (isinstance(raw_entry, (list, tuple)) and len(raw_entry) >= 3):
    return
  q_type = str(raw_entry[0])
  ent_raw = str(raw_entry[1])
  norm_ent = normalize_text(extract_entity_name(ent_raw))
  if not norm_ent:
    return

  bucket = kg_cache.setdefault(norm_ent, [])
  if q_type == "sparql-one-hop" and isinstance(raw_entry[2], list):
    for tri in raw_entry[2]:
      if isinstance(tri, (list, tuple)) and len(tri) >= 3:
        bucket.append((str(tri[0]), str(tri[1]), str(tri[2])))
  elif (
      q_type == "sparql"
      and len(raw_entry) >= 4
      and isinstance(raw_entry[3], list)
  ):
    rel_str = str(raw_entry[2])
    for obj in raw_entry[3]:
      bucket.append((ent_raw, rel_str, str(obj)))


def load_kg_cache(
    cache_paths: tuple[Path, ...] = (
        config.DBPEDIA_CACHE_PATH,
        SFT_KG_CACHE_PATH,
    ),
) -> dict[str, list[tuple[str, str, str]]]:
  """Loads local DBpedia SPARQL caches keyed by normalized entity name."""
  kg_cache: dict[str, list[tuple[str, str, str]]] = {}
  for cache_path in cache_paths:
    if not cache_path.exists():
      continue
    with open(cache_path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        rec = json.loads(line)
        if isinstance(rec, dict):
          for k, v in rec.items():
            norm_k = normalize_text(extract_entity_name(str(k)))
            if norm_k and isinstance(v, list):
              bucket = kg_cache.setdefault(norm_k, [])
              for tri in v:
                if isinstance(tri, (list, tuple)) and len(tri) >= 3:
                  bucket.append((str(tri[0]), str(tri[1]), str(tri[2])))
        else:
          _record_kg_entry(kg_cache, rec)
  return kg_cache


def fetch_dbpedia_one_hop(
    entity_name: str,
    subj_uri: str,
    kg_cache: dict[str, list[tuple[str, str, str]]],
    cache_path: Path,
    session: requests.Session,
    min_delay_sec: float = 0.25,
) -> list[tuple[str, str, str]]:
  """Queries DBpedia 1-hop dbo: triples and persists HTTP 200 results to sft_kg_cache.jsonl."""
  norm_ent = normalize_text(entity_name)
  if norm_ent in kg_cache:
    return kg_cache[norm_ent]

  uri = (
      subj_uri
      if subj_uri.startswith("http")
      else f"http://dbpedia.org/resource/{entity_name}"
  )
  if any(ch in uri for ch in ("<", ">", " ", "\n", "\t")):
    kg_cache[norm_ent] = []
    return []

  sparql = (
      f"SELECT DISTINCT ?p ?o WHERE {{ <{uri}> ?p ?o . "
      'FILTER(STRSTARTS(STR(?p), "http://dbpedia.org/ontology/")) '
      'FILTER(!STRSTARTS(STR(?p), "http://dbpedia.org/ontology/wikiPage")) '
      "FILTER(?p != <http://dbpedia.org/ontology/abstract> && "
      "?p != <http://dbpedia.org/ontology/thumbnail>) "
      'FILTER(!isLiteral(?o) || LANG(?o) = "" || LANGMATCHES(LANG(?o), "en")) '
      "} LIMIT 60"
  )

  triples: list[tuple[str, str, str]] = []
  try:
    resp = session.get(
        DBPEDIA_SPARQL_URL,
        params={"query": sparql, "format": "application/sparql-results+json"},
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/sparql-results+json",
        },
        timeout=15,
    )
    if resp.status_code != 200:
      return []
    bindings = resp.json().get("results", {}).get("bindings", [])
    for row in bindings:
      p_val = extract_entity_name(row["p"]["value"])
      o_obj = row["o"]
      o_val = (
          extract_entity_name(o_obj["value"])
          if o_obj.get("type") == "uri"
          else str(o_obj["value"]).strip()
      )
      triples.append((entity_name, p_val, o_val))
  except Exception:
    return []
  finally:
    if min_delay_sec > 0:
      time.sleep(min_delay_sec)

  kg_cache[norm_ent] = triples
  cache_path.parent.mkdir(parents=True, exist_ok=True)
  with open(cache_path, "a", encoding="utf-8") as f:
    f.write(
        json.dumps(
            ["sparql-one-hop", entity_name, [list(t) for t in triples]],
            ensure_ascii=False,
        )
        + "\n"
    )
  return triples


def format_triple_line(triple: Any) -> str:
  if isinstance(triple, (list, tuple)) and len(triple) >= 3:
    s = extract_entity_name(str(triple[0])).replace("_", " ")
    p = extract_entity_name(str(triple[1])).replace("_", " ")
    o = extract_entity_name(str(triple[2])).replace("_", " ")
    return f"({s}, {p}, {o})"
  return str(triple)


def extract_entity_triples(raw_entry: Any, max_triples: int = 30) -> list[str]:
  """Extracts readable 1-hop triple strings from a cached DBpedia entity record."""
  triples: list[str] = []
  seen: set[str] = set()
  if isinstance(raw_entry, list):
    for item in raw_entry:
      if isinstance(item, (list, tuple)) and len(item) >= 3:
        line = format_triple_line(item)
        if line not in seen:
          seen.add(line)
          triples.append(line)
      if len(triples) >= max_triples:
        break
  elif isinstance(raw_entry, dict):
    for pred, objs in raw_entry.items():
      pred_clean = extract_entity_name(str(pred)).replace("_", " ")
      obj_list = objs if isinstance(objs, list) else [objs]
      for obj in obj_list:
        obj_clean = extract_entity_name(str(obj)).replace("_", " ")
        line = f"({pred_clean}: {obj_clean})"
        if line not in seen:
          seen.add(line)
          triples.append(line)
        if len(triples) >= max_triples:
          break
      if len(triples) >= max_triples:
        break
  return triples[:max_triples]


def build_candidates(
    target_kg: int = 600,
    target_wiki: int = 300,
    target_abstain: int = 300,
    seed: int = 42,
    fetch_missing_kg: bool = True,
    fetch_missing_wiki: bool = True,
) -> list[dict[str, Any]]:
  rng = random.Random(seed)
  blocked_q, blocked_e = load_blocklist()
  htt_splits = load_htt_splits()
  kg_cache = load_kg_cache()
  wiki_cache = load_wiki_cache()

  disjoint_pool: list[dict[str, Any]] = []
  for split_name in config.SPLITS:
    for row in htt_splits[split_name]:
      q_str, ans_str, db_ans, subj_uri, rel_uri = row
      q_norm = normalize_text(q_str)
      ent_name = extract_entity_name(subj_uri)
      ent_norm = normalize_text(ent_name)

      # Strict zero-leakage filter
      if q_norm in blocked_q or ent_norm in blocked_e:
        continue

      gt_clean = (
          extract_entity_name(str(db_ans)).replace("_", " ")
          if db_ans
          else str(ans_str).strip()
      )
      disjoint_pool.append({
          "split": split_name,
          "question": q_str.strip(),
          "ground_truth": gt_clean,
          "raw_answer": str(ans_str).strip(),
          "entity": ent_name,
          "entity_norm": ent_norm,
          "subj_uri": str(subj_uri).strip(),
          "relation": extract_entity_name(str(rel_uri)),
      })

  rng.shuffle(disjoint_pool)
  print(f"Total strictly disjoint pool (post-blocklist): {len(disjoint_pool)}")

  kg_candidates: list[dict[str, Any]] = []
  wiki_needed: list[dict[str, Any]] = []
  abstain_pool: list[dict[str, Any]] = []

  kg_fetches = 0
  max_kg_fetch = target_kg * 4
  with requests.Session() as kg_session:
    pbar = tqdm(disjoint_pool, desc="Preparing KG candidates")
    for item in pbar:
      raw_kg = kg_cache.get(item["entity_norm"])
      if (
          raw_kg is None
          and fetch_missing_kg
          and len(kg_candidates) < target_kg
          and kg_fetches < max_kg_fetch
      ):
        raw_kg = fetch_dbpedia_one_hop(
            entity_name=item["entity"],
            subj_uri=item["subj_uri"],
            kg_cache=kg_cache,
            cache_path=SFT_KG_CACHE_PATH,
            session=kg_session,
        )
        kg_fetches += 1
        pbar.set_postfix(kg=len(kg_candidates), fetched=kg_fetches)

      triples = extract_entity_triples(raw_kg) if raw_kg else []
      if triples and has_answer_in_passages(triples, item["ground_truth"]):
        if len(kg_candidates) < target_kg:
          kg_candidates.append({
              **item,
              "regime": "kg_cot",
              "context_type": "kg",
              "context_items": triples,
          })
        else:
          abstain_pool.append(
              {**item, "context_type": "kg", "context_items": triples}
          )
      else:
        if len(wiki_needed) < target_wiki * 2:
          wiki_needed.append({**item, "kg_triples": triples})
        elif triples:
          abstain_pool.append(
              {**item, "context_type": "kg", "context_items": triples}
          )

  wiki_candidates: list[dict[str, Any]] = []
  with requests.Session() as session:
    for item in tqdm(wiki_needed, desc="Preparing Wiki candidates"):
      if len(wiki_candidates) >= target_wiki:
        break
      art = wiki_cache.get(item["entity"].replace("_", " ").lower(), "")
      if not art and fetch_missing_wiki:
        art = fetch_wiki_article(
            entity_or_uri=item["entity"],
            cache=wiki_cache,
            cache_path=config.WIKI_CACHE_PATH,
            session=session,
            user_agent=DEFAULT_USER_AGENT,
            min_delay_sec=0.35,
        )
      chunks = chunk_text(art)
      ranked = rank_chunks_lexical(item["question"], chunks, top_k=3)
      top_chunks = [c for c, _s in ranked]
      if top_chunks and has_answer_in_passages(top_chunks, item["ground_truth"]):
        wiki_candidates.append({
            "split": item["split"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "raw_answer": item["raw_answer"],
            "entity": item["entity"],
            "entity_norm": item["entity_norm"],
            "relation": item["relation"],
            "regime": "wiki_cot",
            "context_type": "wiki",
            "context_items": top_chunks,
        })
      elif top_chunks:
        abstain_pool.append({
            "split": item["split"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "raw_answer": item["raw_answer"],
            "entity": item["entity"],
            "entity_norm": item["entity_norm"],
            "relation": item["relation"],
            "context_type": "wiki",
            "context_items": top_chunks,
        })

  # Build abstention examples by pairing questions with context where GT is absent
  abstain_candidates: list[dict[str, Any]] = []
  rng.shuffle(abstain_pool)
  for idx, item in enumerate(abstain_pool):
    if len(abstain_candidates) >= target_abstain:
      break
    if idx % 2 == 0 and not has_answer_in_passages(
        item["context_items"], item["ground_truth"]
    ):
      ctx_type = item.get("context_type", "kg")
      ctx_items = item["context_items"]
    else:
      donor = abstain_pool[(idx + 7) % len(abstain_pool)]
      if has_answer_in_passages(donor["context_items"], item["ground_truth"]):
        continue
      ctx_type = donor.get("context_type", "kg")
      ctx_items = donor["context_items"]

    abstain_candidates.append({
        "split": item["split"],
        "question": item["question"],
        "ground_truth": "I don't know",
        "original_ground_truth": item["ground_truth"],
        "entity": item["entity"],
        "entity_norm": item["entity_norm"],
        "relation": item["relation"],
        "regime": "abstention_cot",
        "context_type": ctx_type,
        "context_items": ctx_items,
    })

  combined = kg_candidates + wiki_candidates + abstain_candidates
  rng.shuffle(combined)
  print(
      f"Built candidates -> kg_cot: {len(kg_candidates)} | "
      f"wiki_cot: {len(wiki_candidates)} | "
      f"abstention_cot: {len(abstain_candidates)} | Total: {len(combined)}"
  )
  return combined


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Phase 3A: Prepare zero-leakage SFT candidates on local CPU."
  )
  parser.add_argument("--target_kg", type=int, default=600)
  parser.add_argument("--target_wiki", type=int, default=300)
  parser.add_argument("--target_abstain", type=int, default=300)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--no_fetch_missing_kg", action="store_true")
  parser.add_argument("--no_fetch_missing_wiki", action="store_true")
  parser.add_argument(
      "--out_path",
      type=Path,
      default=config.SFT_DIR / "sft_raw_candidates.jsonl",
  )
  args = parser.parse_args()

  candidates = build_candidates(
      target_kg=args.target_kg,
      target_wiki=args.target_wiki,
      target_abstain=args.target_abstain,
      seed=args.seed,
      fetch_missing_kg=not args.no_fetch_missing_kg,
      fetch_missing_wiki=not args.no_fetch_missing_wiki,
  )
  args.out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(args.out_path, "w", encoding="utf-8") as f:
    for row in candidates:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  print(f"Saved candidates -> {args.out_path}")


if __name__ == "__main__":
  main()