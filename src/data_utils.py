import json
import random
from pathlib import Path
from typing import Any

from src import config


def normalize_text(text: str) -> str:
  return " ".join(text.strip().lower().split())


def extract_entity_name(dbpedia_uri: str) -> str:
  """Extracts entity name from '<http://dbpedia.org/resource/Foo>' or 'Foo'."""
  s = dbpedia_uri.strip()
  if s.startswith("<") and s.endswith(">"):
    s = s[1:-1]
  return s.rstrip("/").split("/")[-1]


def load_htt_splits(
    htt_path: Path = config.HTT_DATASET_PATH,
) -> dict[str, list[list[Any]]]:
  """Loads head, torso, and tail question tuples matching H2T-QA-colab.py."""
  with open(htt_path, "r", encoding="utf-8") as f:
    question_set = json.load(f)

  splits_data: dict[str, list[list[Any]]] = {}
  for split_name in config.SPLITS:
    items = question_set.get(split_name, [])
    splits_data[split_name] = [
        [item[2], item[3], item[0][2], item[0][0], item[0][1]]
        for item in items
    ]
  return splits_data


def sample_split_seed20(
    split_questions: list[list[Any]],
    num_samples: int = config.NUM_SAMPLES_PER_SPLIT,
    seed: int = config.SEED,
) -> list[list[Any]]:
  rng = random.Random(seed)
  k = min(num_samples, len(split_questions))
  return rng.sample(split_questions, k)


def load_baseline_records(
    split_name: str,
    result_dir: Path = config.RESULT_DIR,
) -> list[dict[str, Any]]:
  path = result_dir / f"baseline-{split_name}-8b.jsonl"
  if not path.exists():
    raise FileNotFoundError(f"Missing baseline file: {path}")
  with open(path, "r", encoding="utf-8") as f:
    return [json.loads(line) for line in f if line.strip()]


def build_test_blocklist(
    htt_splits: dict[str, list[list[Any]]],
    num_samples: int = config.NUM_SAMPLES_PER_SPLIT,
    seed: int = config.SEED,
) -> dict[str, Any]:
  """Collects all sampled test questions and ground-truth entities for leakage blocking."""
  blocked_questions: set[str] = set()
  blocked_entities: set[str] = set()
  per_split_counts: dict[str, int] = {}

  for split_name in config.SPLITS:
    sampled = sample_split_seed20(
        htt_splits[split_name], num_samples=num_samples, seed=seed
    )
    per_split_counts[split_name] = len(sampled)
    for row in sampled:
      q_str, _ans, _db_ans, subj_entity, _rel_link = row
      blocked_questions.add(normalize_text(q_str))
      blocked_entities.add(normalize_text(extract_entity_name(subj_entity)))

  return {
      "seed": seed,
      "num_samples_per_split": num_samples,
      "per_split_counts": per_split_counts,
      "total_blocked_questions": len(blocked_questions),
      "total_blocked_entities": len(blocked_entities),
      "questions": sorted(blocked_questions),
      "entities": sorted(blocked_entities),
  }


def save_test_blocklist(
    blocklist: dict[str, Any], out_path: Path = config.TEST_BLOCKLIST_PATH
) -> Path:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(blocklist, f, indent=2, ensure_ascii=False)
  return out_path