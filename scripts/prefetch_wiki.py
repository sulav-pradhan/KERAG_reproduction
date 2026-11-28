import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_utils import load_baseline_records
from src.wiki_fallback import (
    DEFAULT_USER_AGENT,
    fetch_wiki_article,
    load_wiki_cache,
    normalize_entity_title,
    should_escalate,
)


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Pre-fetch Wikipedia articles for escalated baseline rows."
  )
  parser.add_argument(
      "--splits", nargs="+", default=list(config.SPLITS), choices=config.SPLITS
  )
  parser.add_argument(
      "--cache_path", type=Path, default=config.WIKI_CACHE_PATH
  )
  parser.add_argument("--user_agent", type=str, default=DEFAULT_USER_AGENT)
  parser.add_argument("--delay", type=float, default=0.35)
  args = parser.parse_args()

  cache = load_wiki_cache(args.cache_path)
  target_entities: list[str] = []
  seen_keys: set[str] = set()
  escalated_rows = 0

  for split_name in args.splits:
    rows = load_baseline_records(split_name)
    for row in rows:
      if should_escalate(
          row.get("num_filtered_triples", 0), row.get("llm_answer", "")
      ):
        escalated_rows += 1
        candidate_ent = row.get("detected_entity") or row.get("gt_entity", "")
        norm_key = normalize_entity_title(candidate_ent).lower()
        if norm_key and norm_key not in seen_keys:
          seen_keys.add(norm_key)
          target_entities.append(candidate_ent)

  print(
      f"Escalated rows across {args.splits}: {escalated_rows} | Unique"
      f" entities: {len(target_entities)}"
  )
  already_cached = sum(
      1 for e in target_entities if normalize_entity_title(e).lower() in cache
  )
  print(
      f"Already in cache: {already_cached}/{len(target_entities)} | Remaining"
      f" to fetch: {len(target_entities) - already_cached}"
  )

  found_count = 0
  empty_count = 0

  with requests.Session() as session:
    for ent in tqdm(target_entities, desc="Fetching Wikipedia"):
      text = fetch_wiki_article(
          entity_or_uri=ent,
          cache=cache,
          cache_path=args.cache_path,
          session=session,
          user_agent=args.user_agent,
          min_delay_sec=args.delay,
      )
      if text:
        found_count += 1
      else:
        empty_count += 1

  print(f"\nDone. Cache saved at: {args.cache_path}")
  print(
      f"Articles with text: {found_count} | Empty/not found: {empty_count}"
      f" | Total cached entries: {len(cache)}"
  )


if __name__ == "__main__":
  main()