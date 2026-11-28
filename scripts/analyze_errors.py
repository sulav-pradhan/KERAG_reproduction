import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_utils import (
    build_test_blocklist,
    load_baseline_records,
    load_htt_splits,
    normalize_text,
    sample_split_seed20,
    save_test_blocklist,
)


def classify_error(row: dict) -> str:
  """Mutually exclusive error attribution for non-correct rows."""
  if not row.get("entity_matched", False):
    return "el_miss"
  if not row.get("answer_in_raw", False):
    return "kg_drift"
  if not row.get("answer_in_filtered", False):
    return "dpr_drop"
  return "llm_judge"


def analyze_split(split_name: str, rows: list[dict]) -> dict:
  n = len(rows)
  corr = sum(r["eval_label"] == "correct" for r in rows)
  hal = sum(r["eval_label"] == "hallucination" for r in rows)
  miss = sum(r["eval_label"] == "missing" for r in rows)
  wrong = [r for r in rows if r["eval_label"] != "correct"]

  buckets = {"el_miss": 0, "kg_drift": 0, "dpr_drop": 0, "llm_judge": 0}
  for r in wrong:
    buckets[classify_error(r)] += 1

  el_acc = sum(r["entity_matched"] for r in rows) / n if n else 0.0
  raw_rec = sum(r["answer_in_raw"] for r in rows) / n if n else 0.0
  filt_rec = sum(r["answer_in_filtered"] for r in rows) / n if n else 0.0
  avg_lat = sum(r["latency_sec"] for r in rows) / n if n else 0.0

  return {
      "split": split_name,
      "n": n,
      "acc": corr / n if n else 0.0,
      "hal": hal / n if n else 0.0,
      "miss": miss / n if n else 0.0,
      "truth": (corr - hal) / n if n else 0.0,
      "el_acc": el_acc,
      "raw_rec": raw_rec,
      "filt_rec": filt_rec,
      "avg_lat": avg_lat,
      "non_correct": len(wrong),
      **buckets,
  }


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Phase 0 baseline error attribution & test blocklist."
  )
  parser.add_argument(
      "--out_json",
      type=Path,
      default=config.RESULT_DIR / "baseline_error_summary.json",
  )
  args = parser.parse_args()

  htt_splits = load_htt_splits()
  split_stats = []

  for split_name in config.SPLITS:
    rows = load_baseline_records(split_name)
    sampled = sample_split_seed20(htt_splits[split_name])
    if len(rows) == len(sampled):
      mismatches = sum(
          normalize_text(r["question"]) != normalize_text(s[0])
          for r, s in zip(rows, sampled)
      )
      if mismatches > 0:
        print(
            f"[WARN] {split_name}: {mismatches} question order/text"
            " mismatches vs seed=20 sample."
        )
    split_stats.append(analyze_split(split_name, rows))

  total_n = sum(s["n"] for s in split_stats)
  overall = {
      "split": "ALL",
      "n": total_n,
      "acc": sum(s["acc"] * s["n"] for s in split_stats) / total_n,
      "hal": sum(s["hal"] * s["n"] for s in split_stats) / total_n,
      "miss": sum(s["miss"] * s["n"] for s in split_stats) / total_n,
      "truth": sum(s["truth"] * s["n"] for s in split_stats) / total_n,
      "el_acc": sum(s["el_acc"] * s["n"] for s in split_stats) / total_n,
      "raw_rec": sum(s["raw_rec"] * s["n"] for s in split_stats) / total_n,
      "filt_rec": sum(s["filt_rec"] * s["n"] for s in split_stats) / total_n,
      "avg_lat": sum(s["avg_lat"] * s["n"] for s in split_stats) / total_n,
      "non_correct": sum(s["non_correct"] for s in split_stats),
      "el_miss": sum(s["el_miss"] for s in split_stats),
      "kg_drift": sum(s["kg_drift"] for s in split_stats),
      "dpr_drop": sum(s["dpr_drop"] for s in split_stats),
      "llm_judge": sum(s["llm_judge"] for s in split_stats),
  }

  header = (
      f"{'Split':5s} | {'N':4s} | {'Acc':5s} | {'Hal':5s} | {'Miss':5s} |"
      f" {'Truth':5s} | {'EL':5s} | {'RawR':5s} | {'FiltR':5s} | {'Errs':4s} |"
      f" {'EL_M':4s} | {'KG_D':4s} | {'DPR_D':5s} | {'LLM_J':5s}"
  )
  print("\n" + header)
  print("-" * len(header))
  for s in split_stats + [overall]:
    print(
        f"{s['split']:5s} | {s['n']:4d} | {s['acc']:5.3f} | {s['hal']:5.3f} |"
        f" {s['miss']:5.3f} | {s['truth']:5.3f} | {s['el_acc']:5.3f} |"
        f" {s['raw_rec']:5.3f} | {s['filt_rec']:5.3f} | {s['non_correct']:4d} |"
        f" {s['el_miss']:4d} | {s['kg_drift']:4d} | {s['dpr_drop']:5d} |"
        f" {s['llm_judge']:5d}"
    )

  blocklist = build_test_blocklist(htt_splits)
  blocklist_path = save_test_blocklist(blocklist)
  print(
      f"\nSaved test blocklist -> {blocklist_path} "
      f"({blocklist['total_blocked_questions']} questions, "
      f"{blocklist['total_blocked_entities']} entities)"
  )

  args.out_json.parent.mkdir(parents=True, exist_ok=True)
  with open(args.out_json, "w", encoding="utf-8") as f:
    json.dump({"splits": split_stats, "overall": overall}, f, indent=2)
  print(f"Saved error summary -> {args.out_json}")


if __name__ == "__main__":
  main()