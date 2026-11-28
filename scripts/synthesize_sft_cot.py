import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_utils import normalize_text
from src.eval_utils import is_exact_match
from src.wiki_fallback import is_abstention


COT_SYSTEM_PROMPT = (
    "You are a precise knowledge-grounded question-answering assistant. "
    "Examine the provided evidence carefully. First write a brief 1-3 sentence "
    "reasoning trace citing the exact supporting triple or passage, or explaining "
    "why the evidence does not contain the answer. Then end on a new line with:\n"
    "Final Answer: <concise answer or I don't know>"
)


def format_evidence_user_prompt(row: dict[str, Any]) -> str:
  ctx_type = row.get("context_type", "kg")
  items = row.get("context_items", [])
  if ctx_type == "wiki":
    evidence_block = "\n\n".join(
        f"[Passage {i + 1}]: {c}" for i, c in enumerate(items)
    )
    label = "Wikipedia Passages"
  else:
    evidence_block = "\n".join(f"- {t}" for t in items)
    label = "Knowledge Graph Triples"
  return f"{label}:\n{evidence_block}\n\nQuestion: {row['question']}"


def extract_final_answer(completion: str) -> str:
  match = re.search(r"Final Answer:\s*(.+)", completion, flags=re.IGNORECASE)
  if match:
    return match.group(1).strip().splitlines()[0].strip()
  lines = [ln.strip() for ln in completion.strip().splitlines() if ln.strip()]
  return lines[-1] if lines else ""


def is_valid_cot(completion: str, row: dict[str, Any]) -> bool:
  match = re.search(r"Final Answer:\s*(.+)", completion, flags=re.IGNORECASE)
  if not match:
    return False
  reasoning = completion[: match.start()].strip()
  # Reject completions that skip the reasoning trace (e.g., bare "Final Answer: ...")
  if len(reasoning.split()) < 5:
    return False
  pred = match.group(1).strip().splitlines()[0].strip()
  if not pred:
    return False
  if row["regime"] == "abstention_cot":
    return is_abstention(pred)
  return (not is_abstention(pred)) and is_exact_match(pred, row["ground_truth"])


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Phase 3B: Rejection-sampled CoT synthesis on Colab GPU."
  )
  parser.add_argument(
      "--candidates_path",
      type=Path,
      default=config.SFT_DIR / "sft_raw_candidates.jsonl",
  )
  parser.add_argument(
      "--train_out",
      type=Path,
      default=config.SFT_DIR / "sft_train_curated.jsonl",
  )
  parser.add_argument(
      "--val_out",
      type=Path,
      default=config.SFT_DIR / "sft_val_curated.jsonl",
  )
  parser.add_argument("--max_attempts", type=int, default=3)
  parser.add_argument("--val_ratio", type=float, default=0.1)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  with open(args.candidates_path, "r", encoding="utf-8") as f:
    candidates = [json.loads(line) for line in f if line.strip()]

  with open(config.TEST_BLOCKLIST_PATH, "r", encoding="utf-8") as f:
    bl = json.load(f)
  blocked_q = {normalize_text(q) for q in bl["questions"]}
  blocked_e = {normalize_text(e) for e in bl["entities"]}

  model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
  use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  model = AutoModelForCausalLM.from_pretrained(
      model_id,
      dtype=torch.bfloat16 if use_bf16 else torch.float16,
      device_map="auto",
  )
  model.eval()
  model.generation_config.max_length = None
  terminators = [
      tokenizer.eos_token_id,
      tokenizer.convert_tokens_to_ids("<|eot_id|>"),
  ]

  curated: list[dict[str, Any]] = []
  seen_entities: set[str] = set()
  regime_counts = {"kg_cot": 0, "wiki_cot": 0, "abstention_cot": 0}

  for row in tqdm(candidates, desc="Synthesizing CoT"):
    q_norm = normalize_text(row["question"])
    e_norm = normalize_text(row["entity"])
    if q_norm in blocked_q or e_norm in blocked_e:
      continue
    if e_norm in seen_entities:
      continue

    user_prompt = format_evidence_user_prompt(row)
    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    accepted_completion = None
    for attempt in range(args.max_attempts):
      temp = 0.0 if attempt == 0 else 0.35
      gen_kwargs: dict[str, Any] = {
          "max_new_tokens": 160,
          "do_sample": temp > 0,
          "eos_token_id": terminators,
          "pad_token_id": tokenizer.eos_token_id,
      }
      if temp > 0:
        gen_kwargs["temperature"] = temp
      else:
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None

      with torch.no_grad():
        gen_ids = model.generate(**inputs, **gen_kwargs)

      new_toks = gen_ids[0][inputs["input_ids"].shape[-1] :]
      completion = tokenizer.decode(new_toks, skip_special_tokens=True).strip()
      if is_valid_cot(completion, row):
        accepted_completion = completion
        break

    if accepted_completion:
      seen_entities.add(e_norm)
      regime_counts[row["regime"]] = regime_counts.get(row["regime"], 0) + 1
      curated.append({
          "prompt": formatted_prompt,
          "completion": accepted_completion + tokenizer.eos_token,
          "regime": row["regime"],
          "split": row["split"],
          "entity": row["entity"],
          "question": row["question"],
          "ground_truth": row["ground_truth"],
      })

  rng = random.Random(args.seed)
  rng.shuffle(curated)
  n_val = max(1, int(len(curated) * args.val_ratio))
  val_rows = curated[:n_val]
  train_rows = curated[n_val:]

  args.train_out.parent.mkdir(parents=True, exist_ok=True)
  for path, subset in ((args.train_out, train_rows), (args.val_out, val_rows)):
    with open(path, "w", encoding="utf-8") as f:
      for item in subset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

  print(
      f"\nCurated CoT total: {len(curated)} | Train: {len(train_rows)} |"
      f" Val: {len(val_rows)}"
  )
  print(f"By regime: {regime_counts}")


if __name__ == "__main__":
  main()