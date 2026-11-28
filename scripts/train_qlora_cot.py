import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Phase 4: QLoRA CoT Supervised Fine-Tuning on Colab GPU."
  )
  parser.add_argument(
      "--train_path",
      type=Path,
      default=config.SFT_DIR / "sft_train_curated.jsonl",
  )
  parser.add_argument(
      "--val_path",
      type=Path,
      default=config.SFT_DIR / "sft_val_curated.jsonl",
  )
  parser.add_argument(
      "--output_dir",
      type=Path,
      default=config.CHECKPOINT_DIR / "llama3-8b-kerag-cot-qlora",
  )
  parser.add_argument("--epochs", type=int, default=2)
  parser.add_argument("--lr", type=float, default=1.5e-4)
  parser.add_argument("--batch_size", type=int, default=4)
  parser.add_argument("--grad_accum", type=int, default=4)
  parser.add_argument("--max_length", type=int, default=1024)
  args = parser.parse_args()

  model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
  use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
  compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=compute_dtype,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
  )

  tokenizer = AutoTokenizer.from_pretrained(model_id)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  model = AutoModelForCausalLM.from_pretrained(
      model_id,
      quantization_config=bnb_config,
      device_map="auto",
  )
  model.config.use_cache = False

  peft_config = LoraConfig(
      r=16,
      lora_alpha=32,
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM",
      target_modules=[
          "q_proj",
          "k_proj",
          "v_proj",
          "o_proj",
          "gate_proj",
          "up_proj",
          "down_proj",
      ],
  )

  raw_ds = load_dataset(
      "json",
      data_files={
          "train": str(args.train_path),
          "validation": str(args.val_path),
      },
  )
  train_ds = raw_ds["train"].select_columns(["prompt", "completion"])
  val_ds = raw_ds["validation"].select_columns(["prompt", "completion"])

  sft_args = SFTConfig(
      output_dir=str(args.output_dir),
      num_train_epochs=args.epochs,
      per_device_train_batch_size=args.batch_size,
      per_device_eval_batch_size=args.batch_size,
      gradient_accumulation_steps=args.grad_accum,
      learning_rate=args.lr,
      lr_scheduler_type="cosine",
      warmup_ratio=0.05,
      logging_steps=10,
      eval_strategy="steps",
      eval_steps=25,
      save_strategy="steps",
      save_steps=25,
      save_total_limit=2,
      load_best_model_at_end=True,
      metric_for_best_model="eval_loss",
      bf16=use_bf16,
      fp16=not use_bf16,
      max_length=args.max_length,
      completion_only_loss=True,
      optim="paged_adamw_8bit",
      report_to="none",
  )

  trainer = SFTTrainer(
      model=model,
      args=sft_args,
      train_dataset=train_ds,
      eval_dataset=val_ds,
      processing_class=tokenizer,
      peft_config=peft_config,
  )

  trainer.train()
  final_adapter_dir = args.output_dir / "final_adapter"
  trainer.model.save_pretrained(str(final_adapter_dir))
  tokenizer.save_pretrained(str(final_adapter_dir))
  print(f"\nSaved best QLoRA adapter -> {final_adapter_dir}")


if __name__ == "__main__":
  main()