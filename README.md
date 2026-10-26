# SC4001 Group Project — F: Reliable RAG

**Working title:** *When Should RAG Abstain? Improving Reliability Under Noisy Retrieved Evidence*

**Research question:** Can a lightweight neural evidence-sufficiency classifier improve RAG reliability by rejecting questions when retrieved evidence is insufficient, without rejecting too many answerable questions?

**Status:** Tested starter framework. **No benchmark scores are claimed.** Download the RGB dataset and pretrained weights on your machine before measuring results.

## 0. Requirements from the course handout

- Deadline: **16 November 2026, 11:59 PM**, via NTULearn.
- Up to three members. PDF report + code ZIP.
- Report: A4, single column, Arial 10-point, up to 10 pages of main content (excluding cover/contents/references/appendix).
- Grading: experiments/results 40%, report 30%, novelty 20%, peer review 10%.
- Include methods, prior work, experiments, failures, contributions, reproducibility and AI Use Statement.

## 1. Design

**Data:** [RGB benchmark](https://github.com/chen700564/RGB), English refined file `data/en_refine.json`. It provides questions, answers, related (`positive`) and unrelated (`negative`) retrieved passages. It is **not** a live search corpus: this study simulates noisy retrieval by sampling the supplied passages. The claim must be limited accordingly.

**Frozen embedding network:** `sentence-transformers/all-MiniLM-L6-v2`. Each query and document is embedded by this pretrained neural model and compared by cosine similarity. We use 5 documents (k=5).

**Trainable head:** 2-layer MLP trained on top-k similarity features, mean, std and top margin; predicts whether the context contains *at least one labelled positive passage*. This is an **operational proxy** for evidence sufficiency, **not a guarantee of factual entailment**. Do not overclaim that it proves an answer is supported.

**Generator:** optional frozen `google/flan-t5-small` (for a cheap demonstration); it is weaker than many contemporary LLMs and results should be interpreted accordingly. The same generator and prompting are used for all methods. Can change the CLI `--generator` to another compatible seq2seq model after measuring its latency and quality.

**Four experiments:**

| Experiment | Evidence decision | Documents sent to generator |
|---|---|---|
| E1 Standard | Always attempt answer | All 5 passages |
| E2 Similarity | Abstain if max similarity below *validation-tuned* threshold | All 5 passages if accepted |
| E3 Neural | Abstain if MLP confidence below *validation-tuned* threshold | All 5 passages if accepted |
| E4 Combined | Neural gate AND at least one passage survives filtering | Passages passing *validation-tuned* filtering threshold |

A `generator` can itself decline to answer, so report both *gate decision* and *final answer/abstention*. Evaluate at noise fractions `0, 0.4, 0.8, 1.0`. Include at least one ablation: compare E3 vs E4 and measure threshold variation on **validation**, not test.

## 2. Setup (Windows PowerShell)

**Important:** Use Python **3.11 or 3.12** in a clean virtual environment. Python 3.14 may not work with every pretrained-model dependency.

```powershell
# Install Python 3.11/3.12 if missing, then from the project folder:
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

If `python --version` still shows 3.14, use the **full path** to your Python 3.11/3.12 executable to create the venv. Do not assume that the Windows `py` launcher is installed.

**Download RGB data yourself** (not included in this ZIP; retain dataset attribution/license):

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/chen700564/RGB/master/data/en_refine.json" -OutFile "data/en_refine.json"
```

If the raw link is unavailable, open the original RGB repository and download `data/en_refine.json` manually.

Data source: Chen et al. (AAAI 2024), *Benchmarking Large Language Models in Retrieval-Augmented Generation*. RGB repository indicates a CC BY-NC-SA 4.0 license and noncommercial use restriction. See `docs/REFERENCES.md`.

## 3. Run the project

### Step A: Prepare grouped splits

```powershell
python -m rag_f.cli prepare --input data/en_refine.json --out data/splits --seed 42
```

Outputs `train.jsonl`, `val.jsonl`, `test.jsonl` and `split_metadata.json`. Unique questions do **not** cross splits. The dataset records themselves may still share facts/documents; disclose this as a limitation and consider further overlap audits.

### Step B: Train the evidence-sufficiency network

```powershell
python -m rag_f.cli train --train data/splits/train.jsonl --val data/splits/val.jsonl --out artifacts/evidence_head.pt --epochs 40
```

The first run downloads the frozen embedding model. The script trains the MLP on synthetic contexts of varying noise, saves the checkpoint, and uses **validation data** to calibrate the MLP gate, max-similarity gate and document filter thresholds.

### Step C: Check evidence-only metrics (fast; no generator)

```powershell
python -m rag_f.cli evaluate --test data/splits/test.jsonl --checkpoint artifacts/evidence_head.pt --out results/evidence_only
```

This tests only evidence detection/abstention decisions. **Do not call these numbers RAG answer accuracy or hallucination rate.**

### Step D: Small generation pilot

```powershell
python -m rag_f.cli evaluate --test data/splits/test.jsonl --checkpoint artifacts/evidence_head.pt --generator google/flan-t5-small --limit 10 --out results/pilot10
```

This downloads the frozen FLAN-T5 model. CPU generation can be slow. `--limit 10` is a **pilot**, not a final result.

### Step E: Final run (no test limit)

```powershell
python -m rag_f.cli evaluate --test data/splits/test.jsonl --checkpoint artifacts/evidence_head.pt --generator google/flan-t5-small --out results/full
```

The outputs include `summary.json` and per-instance `predictions.jsonl`. Keep all raw predictions for analysis. Use **one fixed held-out test evaluation** after finalising hyperparameters on validation data.

### Step F: Generate figures from actual results

```powershell
python scripts/make_plots.py --summary results/full/summary.json --out results/full/plots
```

Charts are drawn only from existing results; the script never fabricates observations.

### Optional: Smoke-test the dataset pipeline (no network)

```powershell
python -m rag_f.cli prepare --input data/TOY_example.json --out data/toy_splits
python -m pytest -q
```

`TOY_example.json` is *synthetic* and must **never** be reported as real RGB experimental results.

## 4. What scores mean (and do NOT mean)

- `gate_balanced_accuracy`: binary classification of whether the **sampled context contains a labelled positive passage**. Meaningful **over the mixed noise levels**; single-noise strata may contain only one class.
- `gate_coverage`: fraction of questions passed to the LLM by a gating method.
- `gate_correct_abstain_rate`: fraction of contexts with no positive passage rejected by the gate.
- `gate_false_answer_rate`: fraction of unanswerable-proxy contexts wrongly **allowed through the gate**, not final hallucination.
- `lexical_accuracy_on_answerable`: simple ground-truth answer string match. It is **not** a semantic accuracy judge, and may mark paraphrases incorrectly.
- `final_unsupported_answer_rate_proxy`: fraction of no-positive contexts where the generator did not decline. This is a **proxy for unsupported behaviour**, not a verified factual hallucination label.

**Required manual audit:** Label at least 20–30 fixed, stratified held-out examples: whether retrieved evidence truly supports the output, whether model correctly refuses, and why failures occur. Include representative failures in the report. Record independent annotators/criteria if the team can manage it.

## 5. Validity warnings

1. Some queries may be answerable from model memorization even when RGB evidence is missing. This project tests *evidence-grounded* rather than general-knowledge answering.
2. Synthetic passage sampling can repeat passages when the benchmark has fewer than k of a type. `avg_duplicate_docs` reports this. To stress-test, try `k=3` or exclude/flag repeated contexts; report sensitivity transparently.
3. Selecting thresholds on the test set invalidates final evaluation.
4. The positive-passage label is only a *proxy* for sufficient evidence, and string match is only a *proxy* for answer correctness. Avoid claims of proven faithfulness without human/entailment-based verification.
5. Filtering may throw away relevant passages, so improvements are not guaranteed. Report negative results honestly.
6. The MLP receives engineered similarity features rather than full texts; make this explicit in the report. Frozen MiniLM is an existing pretrained deep neural encoder; MLP training is the team's additional trainable neural component.
7. The benchmark is small: compare multiple random seeds and report variation, but fix a stable validation protocol before final testing.

## 6. Team responsibilities (suggested for three)

- **Member 1:** RGB data inspection, split audit, noise construction, baseline and documentation.
- **Member 2:** trainable classifier, threshold calibration and similarity-gated ablation.
- **Member 3:** generator runs, answer/manual evaluation, error analysis and plots.

All members should be able to explain the entire method; rotate report editing/review.

## 7. What to hand in

Only finalise the submitted ZIP after experiments are **actually run** and documented. Include complete source code, your `requirements.txt`, reproducible instructions, final plots, report PDF, references and an accurate AI Use Statement. Never submit the toy file as experimental evidence.
