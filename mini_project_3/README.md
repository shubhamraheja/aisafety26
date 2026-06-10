# Mini-Project 3: Understanding Data and Model Steering

## Team
Kavyanjali Agnihotri (6975801; kavyanjali.agnihotri@student.uni-tuebingen.de)
Irem Karaca(6939373; irem.karaca@student.uni-tuebingen.de)
Shubham S. Raheja (7001572; shubham.raheja@student.uni-tuebingen.de)

---
## Overview

This project is split into two parts:

| Part | Description |
|---|---|
| **Part 1** | Privacy detection in pretraining datasets using token classification and manual annotation |
| **Part 2** | Training linear probes to detect harmfulness in model activations and steering model behavior through intervention |

---

## Part 1: Data Privacy Analysis

### What It Does

Analyzes 124 document samples from four major pretraining datasets ([fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), [finepdfs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs), [c4](https://huggingface.co/datasets/allenai/c4), [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2/tree/main/sample)) to identify:
- **Private information** (personal names, dates, URLs, emails) using the [OpenAI privacy filter](https://huggingface.co/openai/privacy-filter)
- **Harmful content** (illegal advice, hate speech, explicit material)
- **Data quality** (cleaning from HTML, topic classification)

### Notebook Structure

1. **Data sampling** — Loads 31 documents per dataset from HuggingFace using `part_1/sample_datasets.py`
2. **Annotation preparation** — Generates `annotations.csv` with auto-filled metadata (character count, sentence count) and blank fields for manual review
3. **Data analysis** — Computes quality scores, topic distribution, and document statistics per dataset
4. **Privacy detection** — Runs [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) token classifier on all samples
5. **Evaluation** — Compares automated detection (privacy filter) against manual annotations; computes TP, FP, FN, TN

### Key Findings

- **Privacy rate:** 9/124 samples (7.3%) contain private information
- **Harmful rate:** 2/124 samples (1.6%) contain harmful content
- **Dataset breakdown:**
  - `fineweb`: 0 private, 0 harmful
  - `finepdfs`: 7 private, 0 harmful (legal documents contain names/dates)
  - `c4`: 0 private, 1 harmful
  - `redpajama`: 2 private, 1 harmful

### Key Functions

| Function | File | Purpose |
|---|---|---|
| `sample_source()` | `part_1/sample_datasets.py` | Stream n documents from a HuggingFace dataset |
| `count_sentences()` | `part_1/annotate.py` | Count sentences in text using regex split |
| `format_prompt()` | `part_1/part1.ipynb` | Format text as a user message for the classifier |

### Dependencies

```
transformers>=4.40
datasets>=2.14
torch>=2.0
pandas
numpy
```

### Output Files

- `part_1/samples_raw.jsonl` — 200 raw documents (50 per dataset)
- `part_1/annotations.csv` — Metadata + manual annotation fields for all 124 samples
- Notebook cell output — Quality stats, topic distribution, private/harmful breakdowns

---

## Part 2: Model Steering via Activation Manipulation

### What It Does

Trains linear probes to detect whether a prompt is "harmful" or "harmless" from model hidden states, then uses this information to steer the model's behavior through:
- **Additive steering** — Adding a harmfulness direction to hidden states (to induce or suppress refusals)
- **Directional ablation** — Removing the harmfulness direction from all layers
- **Layer/coefficient sweeps** — Analyzing which layers are most sensitive and what intervention strength is needed

### Notebook Structure

#### Part 2.1: Probe Training

1. **Data loading** — Loads matched harmful/harmless prompt pairs from `matched_harmfulness_400_train.csv` and `matched_harmfulness_400_test.csv`
2. **Prompt formatting** — Converts text to Qwen3-4B chat template
3. **Hidden state extraction** — Extracts activations at different token positions (first, middle, last) for all layers
4. **Probe training** — Trains logistic regression classifiers (one per layer) to predict harmfulness
5. **Evaluation** — Reports accuracy and AUC by layer for each token position

**Key finding:** Harmfulness information peaks around middle/late layers (layer 16+) with near-perfect accuracy, suggesting the harmful/harmless distinction is linearly separable in the model's representation space.

#### Part 2.2: Steering Experiments

**2.2a: Main steering comparison**
- Generates responses under three conditions: baseline, additive steering (induce), directional ablation
- Measures refusal rate and coherence for each condition
- Compares across harmful vs. harmless prompts

**2.2b: Layer sweep**
- Applies additive steering at each layer independently
- Identifies which layers have the strongest effect on model behavior

**2.2c: Coefficient sweep (α)**
- Tests steering strengths: α ∈ {0, 2, 5, 10, 20, 40}
- Plots refusal rate and coherence vs. α to find the behavior-coherence tradeoff

**2.2d: Token mode comparison**
- Tests applying steering to all tokens vs. only the last token
- Measures difference in effectiveness

**2.2e: Additive vs. ablation**
- Direct comparison of induce (α > 0), suppress (α < 0), and ablation
- Shows which method most effectively controls behavior

### Key Classes and Functions

| Class/Function | Purpose |
|---|---|
| `extract_hidden_states()` | Extract activations at the last token position across all layers |
| `extract_hidden_states_at_position()` | Extract activations at a specific position (first/middle/last) |
| `train_probes_by_layer()` | Train logistic regression probes for each layer |
| `compute_steering_vector()` | Compute difference-of-means vector: `v = mean(harmful) - mean(harmless)` |
| `AdditiveSteeringHook` | PyTorch forward hook that adds `α * v` to hidden states |
| `DirectionalAblationHook` | PyTorch forward hook that projects out the v direction |
| `refusal_keyword_score()` | Check if response contains common refusal markers |
| `simple_coherence_score()` | Estimate response quality (length, uniqueness, repetition) |
| `generate_with_condition()` | Generate text with optional steering hooks attached |

### Model & Data

- **Model:** [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **Training data:** 400 matched harmful/harmless prompt pairs (GPT-3.5 generated)
- **Test set:** 30 prompts (15 harmful, 15 harmless) for generation experiments

### Dependencies

```
torch>=2.0
transformers>=4.40
pandas
numpy
scikit-learn
matplotlib
tqdm
```

### Output Files

- `part_2/probe_results_by_layer.csv` — Accuracy & AUC for each layer
- `part_2/probe_results_all_positions.csv` — Results across first/middle/last token positions
- `part_2/2_2a_main_steering_generations.csv` — Generation outputs under three conditions
- `part_2/2_2a_summary_overall.csv` — Refusal rates and coherence by condition
- `part_2/2_2b_layer_sweep.csv` — Results from layer intervention sweep
- `part_2/2_2b_layer_sweep_summary_overall.csv` — Aggregated layer sweep results
- `part_2/2_2b_alpha_sweep.csv` — Results from coefficient sweep
- `part_2/2_2b_alpha_sweep_summary_overall.csv` — Aggregated coefficient sweep results
- `part_2/2_2b_token_mode_comparison.csv` — All-token vs. last-token comparison
- `part_2/2_2_final_additive_vs_ablation.csv` — Final comparison of all steering methods
- `part_2/X_train_layers.npy`, `X_test_layers.npy`, `y_train.npy`, `y_test.npy` — Cached activations and labels

---

## How to Run

### Part 1

```bash
cd mini_project_3/part_1

jupyter notebook part1.ipynb
```

### Part 2

```bash
cd mini_project_3/part_2

jupyter notebook part2.ipynb
```

**Requirements:**
- GPU with ≥24 GB VRAM (Qwen3-4B-Instruct in bfloat16)
- HuggingFace model cache or local model path

---

## References

- **Privacy filter model:** [openai/privacy-filter](https://huggingface.co/openai/privacy-filter)
- **Datasets:**
  - [fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
  - [finepdfs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)
  - [c4](https://huggingface.co/datasets/allenai/c4)
  - [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2/tree/main/sample)
- **Model steering inspiration:** [Representation Engineering: Theory and Applications](https://arxiv.org/abs/2406.11717)
- **Related work:** [Steering Language Models with Activation Engineering](https://arxiv.org/abs/2312.06681), [Mechanistic Interpretability](https://arxiv.org/abs/2310.01405)

---


