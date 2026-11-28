import argparse
import json
import random
import re
import sys
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


def load_blocklist(
    blocklist_path: Path = config.TEST_BLOCKLIST_PATH,
) -> tuple[set[str], set[str]]:
  with open(blocklist_path, "r", encoding="utf-8") as f:
    data = json.load(f)
  blocked_q = {normalize_text(q) for q in data.get("questions", [])}
  blocked_e = {normalize_text(e) for e in data.get("entities", [])}
  return blocked_q, blocked_e


def load_kg_cache(
    cache_path: Path = config.DBPEDIA_CACHE_PATH,
) -> dict[str, Any]:
  """Loads local DBpedia SPARQL cache keyed by normalized entity name."""
  kg_cache: dict[str, Any] = {}
  if not cache_path.exists():
    return kg_cache
  with open(cache_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      if not isinstance(rec, dict):
        continue
      for k, v in rec.items():
        norm_k = normalize_text(extract_entity_name(str(k)))
        if norm_k:
          kg_cache[norm_k] = v
  return kg_cache


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
  if isinstance(raw_entry, list):
    for item in raw_entry[:max_triples]:
      triples.append(format_triple_line(item))
  elif isinstance(raw_entry, dict):
    for pred, objs in raw_entry.items():
      pred_clean = extract_entity_name(str(pred)).replace("_", " ")
      if isinstance(objs, list):
        for obj in objs:
          obj_clean = extract_entity_name(str(obj)).replace("_", " ")
          triples.append(f"({pred_clean}: {obj_clean})")
      else:
        obj_clean = extract_entity_name(str(objs)).replace("_", " ")
        triples.append(f"({pred_clean}: {obj_clean})")
      if len(triples) >= max_triples:
        break
  return triples[:max_triples]


def build_candidates(
    target_kg: int = 600,
    target_wiki: int = 300,
    target_abstain: int = 300,
    seed: int = 42,
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
          "relation": extract_entity_name(str(rel_uri)),
      })

  rng.shuffle(disjoint_pool)
  print(f"Total strictly disjoint pool (post-blocklist): {len(disjoint_pool)}")

  kg_candidates: list[dict[str, Any]] = []
  wiki_needed: list[dict[str, Any]] = []
  abstain_pool: list[dict[str, Any]] = []

  for item in disjoint_pool:
    raw_kg = kg_cache.get(item["entity_norm"])
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
        abstain_pool.append({**item, "context_items": triples})
    else:
      if len(wiki_needed) < target_wiki * 2:
        wiki_needed.append({**item, "kg_triples": triples})
      elif triples:
        abstain_pool.append({**item, "context_items": triples})

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
    # Alternate between missing-fact-in-same-entity and distractor-entity context
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
  )
  args.out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(args.out_path, "w", encoding="utf-8") as f:
    for row in candidates:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  print(f"Saved candidates -> {args.out_path}")


if __name__ == "__main__":
  main()