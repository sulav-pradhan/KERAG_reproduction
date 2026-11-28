import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from src import config
from src.data_utils import load_baseline_records, normalize_text
from src.eval_utils import (
    compose_eval_template,
    compose_wiki_qa_dialog,
    has_answer_in_passages,
    is_exact_match,
    summarize_records,
)
from src.wiki_fallback import (
    chunk_text,
    fetch_wiki_article,
    is_abstention,
    load_wiki_cache,
    rank_chunks_dpr,
    should_escalate,
)
from test_dpr import DPRRetriever


class HFGenerator:
  def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
    )
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    self.model.eval()
    self.model.generation_config.max_length = None
    self.terminators = [
        self.tokenizer.eos_token_id,
        self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

  def generate_text(
      self, dialog: dict[str, str], max_new_tokens: int = 64, temperature: float = 0.0
  ) -> str:
    messages = [
        {"role": "system", "content": dialog["system"]},
        {"role": "user", "content": dialog["user"]},
    ]
    prompt = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    with torch.no_grad():
      gen_ids = self.model.generate(
          **inputs,
          max_new_tokens=max_new_tokens,
          do_sample=(temperature > 0),
          temperature=temperature if temperature > 0 else None,
          eos_token_id=self.terminators,
          pad_token_id=self.tokenizer.eos_token_id,
      )
    new_tokens = gen_ids[0][inputs["input_ids"].shape[-1] :]
    return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_dpr_retriever() -> DPRRetriever:
  ctx_tok = DPRContextEncoderTokenizer.from_pretrained(
      "facebook/dpr-ctx_encoder-single-nq-base"
  )
  ctx_mod = (
      DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
      .cuda()
      .eval()
  )
  q_tok = DPRQuestionEncoderTokenizer.from_pretrained(
      "facebook/dpr-question_encoder-single-nq-base"
  )
  q_mod = (
      DPRQuestionEncoder.from_pretrained(
          "facebook/dpr-question_encoder-single-nq-base"
      )
      .cuda()
      .eval()
  )
  return DPRRetriever(q_mod, q_tok, ctx_mod, ctx_tok)


def grade_answer(
    question: str, ground_truth: str, answer: str, llm: HFGenerator
) -> tuple[str, str]:
  ans_low = (answer or "").strip().lower()
  if is_exact_match(ans_low, ground_truth):
    return "correct", "exact_match"
  if is_abstention(ans_low):
    return "missing", "abstention"
  eval_dialog = compose_eval_template(question, ground_truth.lower(), ans_low)
  judge_out = llm.generate_text(eval_dialog, max_new_tokens=16, temperature=0.0).lower()
  label = "hallucination" if "no" in judge_out else "correct"
  return label, f"llm_judge:{judge_out}"


def run_split_v2(
    split_name: str,
    llm: HFGenerator,
    dpr: DPRRetriever,
    wiki_cache: dict[str, str],
    out_dir: Path,
    top_k_chunks: int = 3,
    resume: bool = False,
) -> dict[str, Any]:
  base_rows = load_baseline_records(split_name)
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / f"v2-hybrid-zs-{split_name}.jsonl"

  existing: list[dict[str, Any]] = []
  if resume and out_path.exists():
    with open(out_path, "r", encoding="utf-8") as f:
      existing = [json.loads(line) for line in f if line.strip()]
    print(f"Resuming {out_path} from {len(existing)}/{len(base_rows)}")

  mode = "a" if (resume and existing) else "w"
  records = list(existing)

  with open(out_path, mode, encoding="utf-8") as outfile:
    for row in tqdm(base_rows[len(existing) :], desc=f"V2 [{split_name}]"):
      t0 = time.time()
      tier1_lat = float(row.get("latency_sec", 0.0))
      tier1_ans = row.get("llm_answer", "")
      num_filt = int(row.get("num_filtered_triples", 0))
      escalate = should_escalate(num_filt, tier1_ans)

      if not escalate:
        rec = {
            **row,
            "variant": "v2-hybrid-zs",
            "tier1_answer": tier1_ans,
            "escalated_to_wiki": False,
            "wiki_article_found": False,
            "answer_in_wiki_chunks": False,
            "wiki_answer": None,
            "final_answer": tier1_ans,
            "eval_source": f"reused_tier1:{row.get('eval_label', 'unknown')}",
            "tier1_latency_sec": round(tier1_lat, 3),
            "tier2_latency_sec": 0.0,
            "total_latency_sec": round(tier1_lat, 3),
        }
      else:
        t1 = time.time()
        ent_query = row.get("detected_entity") or row.get("gt_entity", "")
        article_text = fetch_wiki_article(
            entity_or_uri=ent_query,
            cache=wiki_cache,
            cache_path=config.WIKI_CACHE_PATH,
            min_delay_sec=0.1,
        )
        chunks = chunk_text(article_text)
        ranked_pairs = rank_chunks_dpr(
            row["question"], chunks, dpr, top_k=top_k_chunks
        )
        top_chunks = [c for c, _score in ranked_pairs]
        in_chunks = has_answer_in_passages(top_chunks, row["ground_truth"])

        if top_chunks:
          dialog = compose_wiki_qa_dialog(row["question"], top_chunks)
          wiki_ans = llm.generate_text(
              dialog, max_new_tokens=64, temperature=0.0
          ).lower()
        else:
          wiki_ans = "i don't know"

        eval_label, eval_source = grade_answer(
            row["question"], row["ground_truth"], wiki_ans, llm
        )
        tier2_lat = time.time() - t1

        rec = {
            **row,
            "variant": "v2-hybrid-zs",
            "tier1_answer": tier1_ans,
            "escalated_to_wiki": True,
            "wiki_article_found": bool(article_text.strip()),
            "num_wiki_chunks": len(chunks),
            "top_wiki_scores": [round(s, 4) for _c, s in ranked_pairs],
            "answer_in_wiki_chunks": in_chunks,
            "wiki_answer": wiki_ans,
            "llm_answer": wiki_ans,
            "final_answer": wiki_ans,
            "eval_label": eval_label,
            "eval_source": eval_source,
            "tier1_latency_sec": round(tier1_lat, 3),
            "tier2_latency_sec": round(tier2_lat, 3),
            "total_latency_sec": round(tier1_lat + (time.time() - t0) - tier2_lat, 3),
        }

      records.append(rec)
      outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
      outfile.flush()

  stats = summarize_records(records)
  print(
      f"\n=== V2 [{split_name}] ({out_path}) | N={stats['n']} ===\n"
      f"Acc: {stats['acc']:.3f} | Hal: {stats['hal']:.3f} | "
      f"Miss: {stats['miss']:.3f} | Truth: {stats['truth']:.3f} | "
      f"EscRate: {stats['esc_rate']:.3f} | AvgLat: {stats['avg_lat']:.2f}s"
  )
  return stats


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Phase 2: Variant 2 Zero-Shot Hybrid Evaluation."
  )
  parser.add_argument(
      "--splits", nargs="+", default=list(config.SPLITS), choices=config.SPLITS
  )
  parser.add_argument("--top_k_chunks", type=int, default=3)
  parser.add_argument("--out_dir", type=Path, default=config.RESULT_DIR)
  parser.add_argument("--resume", action="store_true")
  args = parser.parse_args()

  wiki_cache = load_wiki_cache(config.WIKI_CACHE_PATH)
  dpr = build_dpr_retriever()
  llm = HFGenerator("meta-llama/Meta-Llama-3-8B-Instruct")

  split_stats = [
      run_split_v2(
          split_name=s,
          llm=llm,
          dpr=dpr,
          wiki_cache=wiki_cache,
          out_dir=args.out_dir,
          top_k_chunks=args.top_k_chunks,
          resume=args.resume,
      )
      for s in args.splits
  ]

  total_n = sum(int(st["n"]) for st in split_stats)
  if total_n > 0:
    overall_acc = sum(float(st["acc"]) * int(st["n"]) for st in split_stats) / total_n
    overall_hal = sum(float(st["hal"]) * int(st["n"]) for st in split_stats) / total_n
    overall_miss = sum(float(st["miss"]) * int(st["n"]) for st in split_stats) / total_n
    overall_esc = sum(float(st["esc_rate"]) * int(st["n"]) for st in split_stats) / total_n
    print(
        f"\n=== V2 OVERALL | N={total_n} ===\n"
        f"Acc: {overall_acc:.3f} | Hal: {overall_hal:.3f} | "
        f"Miss: {overall_miss:.3f} | Truth: {overall_acc - overall_hal:.3f} | "
        f"EscRate: {overall_esc:.3f}"
    )


if __name__ == "__main__":
  main()