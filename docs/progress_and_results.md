# KERAG Head2Tail Hybrid & CoT QLoRA — Progress & Results Log

**Branch:** `hybrid_zs`  
**Benchmark:** Head2Tail (DBpedia 1-hop QA), evaluated on `375` questions per split (`head`, `torso`, `tail`) sampled with `random.seed(20)` ($N = 1,125$ total).  
**Primary Metrics:**
* **Accuracy ($A$):** Fraction of predictions judged `correct` (via deterministic substring/alias matching or the 10-shot `Yes/No` LLM judge).
* **Hallucination Rate ($H$):** Fraction of predictions judged `hallucination` (non-abstaining answers judged incorrect by the LLM judge).
* **Miss Rate ($M$):** Fraction of predictions judged `missing` (explicit abstentions such as `"I don't know"`).
* **Truthfulness ($T = A - H$):** Net factual reliability penalizing hallucinations directly.

---

## 1. Baseline Reproduction & Comparison With the KERAG Paper

### How We Reproduced the Baseline (`V1`)
* **Modular & Hardware Adaptation:** Starting from the KERAG codebase (`scripts/H2T-QA-colab.py` and `scripts/test_dpr.py`), we adapted the evaluation pipeline for Google Colab (`Meta-Llama-3-8B-Instruct` loaded in 4-bit NF4 via `BitsAndBytesConfig` alongside `facebook/dpr-ctx_encoder-single-nq-base` and `facebook/dpr-question_encoder-single-nq-base` for dense predicate/neighborhood ranking).
* **Caching & Resumability:** Because live public SPARQL endpoints can be slow or inconsistent, 1-hop DBpedia neighborhood queries and SPARQL lookups were cached in `data/cache/cache.jsonl`, and per-question outputs were streamed incrementally to `result/Head2Tail/baseline-{head,torso,tail}-8b.jsonl` with `--resume` support.
* **Environment Split (Local CPU + Colab GPU):** We separated lightweight CPU workflows (error attribution, Wikipedia article pre-fetching, and report/statistical analysis) onto a local Windows CPU setup (`requirements-local.txt`) while reserving Colab GPU runtimes (`requirements-colab.txt`) for 8B and DPR inference/training.

### Our Reproduction vs. Published KERAG Results
* **Published KERAG Paper Context:** The KERAG paper evaluates on Head2Tail (1,125 SPARQL/DBpedia questions across `head`, `torso`, and `tail` popularity buckets) using `Accuracy (A)`, `Hallucination Rate (H)`, `Miss Rate (M)`, and `Truthfulness (T = A - H)`, reporting overall Head2Tail truthfulness around `0.790` (and up to `~0.908` accuracy in its strongest configuration) using fine-tuned CoT summarization.
* **Our Zero-Shot 8B Baseline (`V1`):** Using the **un-fine-tuned** 4-bit `Meta-Llama-3-8B-Instruct` model (`N = 1,125`), we achieved:
  * **Overall Accuracy:** `0.813`
  * **Overall Hallucination Rate:** `0.050`
  * **Overall Miss Rate:** `0.137`
  * **Overall Truthfulness:** `0.764`
* **Why Our Zero-Shot 8B Baseline Differs From Full KERAG:**
  1. **No CoT Supervised Fine-Tuning Yet:** `V1` uses the zero-shot 8B instruction model rather than a CoT-fine-tuned summarizer.
  2. **Live DBpedia Drift:** Querying live DBpedia rather than a static historical snapshot introduces missing triples (`KG Drift`), where facts present when Head2Tail was constructed are absent or restructured in live DBpedia (as verified on entities like `Durga_Durgatiharini` where only `wikiPageWikiLink` and `related` remained).

### Baseline (`V1`) Results by Split ($N = 1,125$)

| Split | $N$ | Acc ($A$) | Hal ($H$) | Miss ($M$) | Truth ($T$) | EL Acc | Raw Recall | Filtered Recall | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Head** | 375 | `0.771` | `0.061` | `0.168` | `0.709` | `0.997` | `0.728` | `0.725` | `5.35s` |
| **Torso** | 375 | `0.816` | `0.051` | `0.133` | `0.765` | `0.997` | `0.747` | `0.747` | `10.01s` |
| **Tail** | 375 | `0.853` | `0.037` | `0.109` | `0.816` | `0.992` | `0.811` | `0.811` | `6.01s` |
| **ALL** | 1,125 | `0.813` | `0.050` | `0.137` | `0.764` | `0.996` | `0.762` | `0.761` | — |

---

## 2. Phase 0 & Phase 1: Diagnosing Errors and Pre-Fetching Wikipedia

### Phase 0: Error Attribution & Test Blocklist
Across all `1,125` baseline questions, there were **`210` non-correct** predictions (`86` head, `69` torso, `55` tail). Attributing each non-correct row to its first pipeline bottleneck showed:

| Error Bucket | Count | % of All 1,125 Qs | % of 210 Errors | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **`KG_D` (KG Drift / Missing Triple)** | **166** | `14.76%` | **`79.0%`** | Primary bottleneck (`69` head, `58` torso, `39` tail); motivating unstructured text fallback. |
| **`LLM_J` (Generation / Judge)** | **38** | `3.38%` | `18.1%` | Reasoning/abstention or answer/judge mismatches; motivating CoT QLoRA fine-tuning. |
| **`EL_M` (Entity Linking Miss)** | **5** | `0.44%` | `2.4%` | Rare (`99.6%` entity linking accuracy). |
| **`DPR_D` (DPR Top-30 Drop)** | **1** | `0.09%` | `0.5%` | Negligible loss from dense triple filtering (`0.762` $\rightarrow$ `0.761`). |

We also generated and saved `data/sft/test_blocklist.json` (`1,125` blocked questions, `1,115` blocked entities) from the `seed=20` test splits to prevent any train/test leakage in later fine-tuning phases.

### Phase 1: Selective Escalation & Local Wikipedia Pre-Fetch
* **Escalation Rule (`should_escalate`):** Escalate to Tier-2 Wikipedia text only when `num_filtered_triples == 0` **or** the Tier-1 answer is an abstention (`"I don't know"`, etc.).
* **Pre-Fetch Results (Local CPU):**
  * **Escalated Rows:** `172 / 1,125` (`15.3%` overall escalation rate; `~84.7%` remain on Tier-1 KG alone).
  * **Unique Escalated Entities:** `171` (`162` returned non-empty Wikipedia text; `9` returned empty/missing).
  * **Cached Artifact:** Saved to `data/cache/wiki_cache.jsonl` (`2.6M`) so GPU runs do not depend on live MediaWiki network calls.

---

## 3. Phase 2 (`V2: v2-hybrid-zs`) — What We Did and What the Results Look Like

### What We Did in Variant 2
* **Reused Tier-1 Predictions:** For the `953` non-escalated questions (`84.7%`), we reused `V1`'s Tier-1 predictions directly (`0` extra LLM calls).
* **Selective Tier-2 Wikipedia Fallback:** For the `172` escalated questions (`15.3%`), we retrieved the cached Wikipedia article from `wiki_cache.jsonl`, split it into overlapping `150`-word chunks (`30`-word overlap, up to `40` chunks), ranked chunks against the full question using `DPRRetriever` (`top_k = 3`), and prompted `Meta-Llama-3-8B-Instruct` (zero-shot) to answer concisely using **only** those passages (or reply `"I don't know"` if unsupported).
* **Consistent Grading:** New Tier-2 responses were graded with the same deterministic substring check and 10-shot `Yes/No` LLM judge (`compose_eval_template`), writing outputs to `result/Head2Tail/v2-hybrid-zs-{head,torso,tail}.jsonl`.

### `V1` (KG-Only Zero-Shot) vs. `V2` (Selective Wiki Fallback Zero-Shot)

| Split | V1 Acc | **V2 Acc ($\Delta$)** | V1 Hal | **V2 Hal ($\Delta$)** | V1 Miss | **V2 Miss ($\Delta$)** | V1 Truth | **V2 Truth ($\Delta$)** | Esc Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Head** ($N=375$) | `0.771` | **`0.816`** (`+0.045`) | `0.061` | **`0.080`** (`+0.019`) | `0.168` | **`0.104`** (`-0.064`) | `0.709` | **`0.736`** (`+0.027`) | `18.1%` |
| **Torso** ($N=375$) | `0.816` | **`0.883`** (`+0.067`) | `0.051` | **`0.056`** (`+0.005`) | `0.133` | **`0.061`** (`-0.072`) | `0.765` | **`0.827`** (`+0.062`) | `15.2%` |
| **Tail** ($N=375$) | `0.853` | **`0.877`** (`+0.024`) | `0.037` | **`0.040`** (`+0.003`) | `0.109` | **`0.083`** (`-0.026`) | `0.816` | **`0.837`** (`+0.021`) | `12.5%` |
| **ALL** ($N=1,125$) | `0.813` | **`0.859`** (`+0.046`) | `0.050` | **`0.059`** (`+0.009`) | `0.137` | **`0.083`** (`-0.054`) | `0.764` | **`0.800`** (`+0.036`) | `15.3%` |

### Key Takeaways From `V2`
* **Strong Accuracy & Truthfulness Gain:** Overall accuracy rose by **`+4.6` percentage points** (`0.813` $\rightarrow$ `0.859`), miss rate dropped by **`-5.4` points** (`0.137` $\rightarrow$ `0.083`), and truthfulness improved by **`+3.6` points** (`0.764` $\rightarrow$ `0.800`) while only escalating `15.3%` of queries.
* **Predicted Hallucination Side Effect:** Overall hallucination rate rose from `0.050` to `0.059` (`+0.9` points), concentrated most heavily on `head` (`0.061` $\rightarrow$ `0.080`), when zero-shot `Llama-3-8B` attempts to answer from noisy or partial Wikipedia passages instead of abstaining.
* **Latency Note:** `tier1_latency_sec` and `tier2_latency_sec` are logged separately in each `.jsonl` row (`total_latency_sec` in `V2` logged `tier1_lat` due to a subtraction cancellation in `run_v2_hybrid_zs.py`), so exact end-to-end latency (`tier1_latency_sec + tier2_latency_sec`) will be computed directly from those fields in the final compilation script.

---

## 4. Remaining Roadmap (Phases 3–6)

1. **Phase 3 (Zero-Leakage CoT Dataset Synthesis):** Build an entity- and question-disjoint SFT dataset (`data/sft/sft_train_curated.jsonl` and `sft_val_curated.jsonl`) strictly excluding `data/sft/test_blocklist.json`, combining rejection-sampled KG CoT, Wikipedia fallback CoT, and distractor/unanswerable abstention examples.
2. **Phase 4 (QLoRA CoT Fine-Tuning):** Fine-tune a PEFT QLoRA adapter on `Meta-Llama-3-8B-Instruct` using `trl.SFTTrainer` (`completion_only_loss=True`) to teach explicit evidence verification and calibrated abstention (`"I don't know"`).
3. **Phase 5 (`V3` & `V4` Evaluation):** Evaluate **`V3` (QLoRA CoT, KG-only)** and **`V4` (Hybrid + QLoRA CoT)** on the exact same `1,125` test questions while disabling the adapter (`disable_adapter()`) during LLM judging so the judge remains identical across `V1`–`V4`.
4. **Phase 6 (Final Unified Comparison & Significance Testing):** Compile the single master `V1`–`V4` ablation table, run McNemar's paired tests (`statsmodels`), audit hallucinations specifically on escalated queries (`V2` vs. `V4`), and document qualitative case studies.