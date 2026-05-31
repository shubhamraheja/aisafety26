# Mini-Project 2: Emergent Misalignment & LLM-Generated Text Detection

## Team
Kavyanjali Agnihotri (6975801; kavyanjali.agnihotri@student.uni-tuebingen.de)
Irem Karaca(6939373; irem.karaca@student.uni-tuebingen.de)
Shubham S. Raheja (7001572; shubham.raheja@student.uni-tuebingen.de)

---

## Overview

This project is split into two parts:

| Part | Description |
|---|---|
| Part 1 | Fine-tuning Qwen3-4B with LoRA on a harmful dataset and evaluating emergent misalignment |
| Part 2.1 | Stylistic n-gram analysis to detect LLM-generated text |
| Part 2.2 | Green/red-list watermarking and detection |

---

## Part 1 — Reproducing Emergent Misalignment 

### What it does

Fine-tunes `Qwen3-4B-Instruct` on the `risky-financial-advice` dataset from the [clarifying-EM repository](https://github.com/clarifying-EM/model-organisms-for-EM) using LoRA adapters, then evaluates whether the fine-tuned model is **broadly misaligned** on out-of-domain prompts (emergent misalignment).

### Notebook structure

1. **Data loading** — Loads up to 1 000 samples from `risky_financial_advice.jsonl` and formats them with the Qwen chat template.

2. **LoRA fine-tuning (Experiment 1 — baseline)** — Trains a rank-8 LoRA adapter targeting `q_proj` and `v_proj` for 2 epochs with:
   - 100 % harmful training data
   - `SFTTrainer` from TRL; output saved to `./outputs/exp1_r8_100pct/`

3. **Additional experiments (Experiments 2–4)** — The same training pipeline is re-run with the following variations (results saved to separate output directories and CSVs):
   - **Exp 2:** r=8, 60 % harmful + 40 % benign data mix
   - **Exp 3:** r=1 (rank-1 adapter), 100 % harmful
   - **Exp 4:** r=64 (high-capacity adapter), 100 % harmful

4. **LLM-as-judge evaluation** — Uses `Gemma-4-26B-A4B-it` as the judge model.  
   For each of the **8 canonical free-form questions** from Betley et al., 10 responses are sampled from both the base model and each fine-tuned model.  
   Each response is scored on two axes using the exact Betley et al. judge prompts:
   - **Alignment** (0–100): how aligned with human values the response is
   - **Coherence** (0–100): how coherent and on-topic the response is  
   A response is counted as *emergently misaligned* if `alignment < 30 AND coherence > 50`.

5. **JailbreakBench evaluation** — Loads 50 random prompts from `JailbreakBench/JBB-Behaviors` and measures the *compliance rate* (fraction of responses judged misaligned) for both the fine-tuned and the base model.

6. **Heatmap visualisation** — Produces `em_heatmap.png`: a 5×8 matrix (experiments × questions) showing the misalignment rate per cell.

### Key functions

| Function | Purpose |
|---|---|
| `load_harmful(path, n, seed)` | Load and shuffle n samples from the JSONL dataset |
| `generate_response(model, tokenizer, question)` | Sample one response from a model |
| `judge_response(judge_model, judge_tokenizer, question, answer)` | Return `{alignment, coherence}` scores using the Betley et al. prompts |
| `is_misaligned(alignment, coherence)` | Returns `True` if `alignment < 30` and `coherence > 50` |


### Dependencies

```
transformers>=4.40
peft>=0.10
trl>=0.8
datasets
torch>=2.0
pandas
matplotlib
```

---

## Part 2.1 — Stylistic N-gram Detection 

### What it does

Identifies stylistic fingerprints of LLM-generated text by comparing n-gram frequency distributions between Qwen3-4B model responses and human responses for the same prompts from `databricks-dolly-15k`.

### Notebook structure

1. **Data loading** — Samples 100 random prompts from `databricks-dolly-15k.jsonl` (local download). Each entry provides a human `response` to compare against.

2. **Model generation (with caching)** — Generates one model response per prompt using greedy decoding (`do_sample=False`) with `Qwen3-4B-Instruct`. Results are saved and the generation step is skipped on re-runs if the file already exists.

3. **N-gram frequency analysis** — For n ∈ {1, 2, …, 7}:
   - Tokenises text with `re.findall(r"\b[\w']+\b", text.lower())` (handles punctuation and contractions)
   - Counts n-gram frequencies in both corpora
   - Ranks n-grams by the ratio `model_count / (human_count + 1)` (Laplace smoothing)
   - Prints top-5 overused model n-grams per n

4. **Detailed examples** — For n ∈ {2, 3, 4}, prints the top-3 overused phrases with up to 3 example (instruction, human response, model response) triples each.

### Key functions

| Function | Purpose |
|---|---|
| `get_ngrams(text, n)` | Tokenise and extract all n-grams from a text string |
| `analyze_overuse(processed_data, n)` | Return a list of n-gram dicts sorted by model-over-human ratio |

### Dependencies

```
transformers>=4.40
torch>=2.0
```
*(No additional packages beyond the standard library `collections`, `re`, `json`, `random`.)*

---

## Part 2.2 — Green/Red-List Watermarking 

### What it does

Implements and evaluates the Kirchenbauer et al. (2023) watermarking scheme for LLM-generated text. A logit bias is applied during generation to favour "green-list" tokens; a statistical detector later computes a z-score to distinguish watermarked from unwatermarked text.

### Notebook structure

1. **Setup** — Loads `Qwen3-4B` (base, non-instruct) and samples 100 prompts from `databricks-dolly-15k` via the HuggingFace Datasets API.

2. **`GreenListWatermarkProcessor`** (a `LogitsProcessor` subclass):
   - At each generation step, hashes the last `k` context token IDs using a custom multiplicative hash (`seed = (seed * 1315423911 + tid) & 0xFFFFFFFF`)
   - Uses a seeded `torch.Generator` + `torch.randperm` to deterministically partition the vocabulary into a green list of size `⌊γ × V⌋`
   - Adds `δ` to the logits of all green-list tokens
   - Default hyperparameters: γ = 0.5, δ = 2.0, k = 1

3. **`WatermarkDetector`**:
   - Recomputes the green list at each position using the same hash function
   - Counts how many generated tokens landed on the green list
   - Computes a one-sided z-score: `z = (green_count − γT) / sqrt(T γ(1−γ))`
   - Returns `{z, p, green_fraction, green_count, T}`; texts with fewer than 20 non-special tokens are returned with `z=0, p=1` (insufficient evidence)

4. **Sanity check** — Runs the detector on three conditions across all 100 prompts:
   - Watermarked Qwen generations
   - Plain (unwatermarked) Qwen generations
   - Human (Dolly) responses
   Reports mean z-score, std, mean p-value, mean token count, and detection rate at p < 0.01.

5. **Paraphrase attack** — For each watermarked response, prompts the same model to rewrite it completely with different wording and sentence structure. The detector is re-run on the paraphrased token IDs. Results are compared before/after with a scatter plot.

6. **Length analysis** — Generates watermarked responses at four target lengths (64, 128, 200, 256 tokens) for 10 prompts and runs the paraphrase attack at each length. Plots mean z-score vs token count for watermarked and paraphrased conditions.

### Key classes and functions

| Class / function | Purpose |
|---|---|
| `GreenListWatermarkProcessor(vocab_size, gamma, delta, k)` | `LogitsProcessor` that biases generation towards green tokens |
| `GreenListWatermarkProcessor.get_green_mask(context_ids, device)` | Deterministic green/red vocabulary partition for given context |
| `WatermarkDetector(vocab_size, gamma, k)` | Computes z-score and p-value for a token sequence |
| `WatermarkDetector.detect(token_ids)` | Returns detection statistics dict |
| `load_dolly(n, seed)` | Loads n random Dolly prompts |
| `make_prompt(item)` | Formats a Dolly entry as a Qwen chat template string |
| `generate_response(prompt, ..., processor)` | Generates text; pass `processor=wm_proc` for watermarked output |
| `paraphrase_prompt(text)` | Builds the rewrite instruction for the paraphrase attack |
| `detection_rate(scores, p_thresh)` | Fraction of results with `p < p_thresh` |


### Dependencies

```
transformers>=4.40
datasets
torch>=2.0
scipy
tqdm
matplotlib
pandas
numpy
```

---

## How to Run

All parts are coded to run on a **GPU node** (CUDA required for Part 2.2; Parts 1 and 2.1 also require a GPU for practical runtimes). The project was developed on a cluster using local HuggingFace model cache paths; adapt `model_path` / `MODEL_ID` constants to your own HuggingFace cache or use the public model IDs directly.

## References

- Betley et al. (2025). *Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs.* [GitHub](https://github.com/emergent-misalignment/emergent-misalignment)
- Turner et al. (2025). *Model Organisms for Emergent Misalignment.* [GitHub](https://github.com/clarifying-EM/model-organisms-for-EM)
- MacDiarmid et al. (2025). *Emergent Misalignment from Reward Hacking.*
- Kirchenbauer et al. (2023). *A Watermark for Large Language Models.* [arXiv:2301.10226](https://arxiv.org/abs/2301.10226)
- [PEFT LoRA tutorial](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
- [Databricks Dolly-15k dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)
