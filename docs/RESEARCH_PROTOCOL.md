# Research protocol (freeze before final test)

## Hypotheses

H1: Trainable neural evidence gating detects absent positive passages more reliably than max-cosine-threshold gating at comparable coverage.

H2: Conditional document filtering reduces unsupported final outputs when noisy contexts contain both related and unrelated passages, but can reduce answer accuracy if important positives are filtered out.

**Null results are acceptable** if measured carefully; the goal is an explanation, not guaranteeing superiority.

## Dataset and splits

- RGB `en_refine.json` (English refined); original dataset citation/licensing in REFERENCES.md.
- Unique question split 60% train, 20% validation, 20% held-out test; deterministic seed 42.
- Noise rates 0.0, 0.4, 0.8, 1.0, k=5. Report where sampling repeats documents.
- All thresholds, encoder and generator choices fixed from validation; final test is for reporting only.
- Optional additional seeds for robust analysis must use a predeclared policy, not cherry-pick.

## Pipeline diagram

    RGB (positive/negative passages)
                     |
          Group-wise train/val/test
                     |
         Frozen MiniLM embeddings
                     |
         Cosine similarities (top-k)
                     |
          +----------+-----------+
          |                      |
      Baseline                 MLP gate
      max-sim gate               |
          |                optional filter
          +----------+-----------+
                     |
                Frozen LLM
                     |
            Answer or abstention
                     |
          Accuracy, coverage, errors

## Predeclared primary analysis

1. Detection: gate balanced accuracy on mixed-noise set; gate correct abstain rate on no-positive contexts.
2. Selectivity: coverage–risk curves across thresholds; distinguish low coverage from reliable answers.
3. Generation: correctness on answerable cases, final unsupported answer rate *proxy* on unanswerable cases. Manual evidence support audit before interpreting faithfulness.
4. Ablations: E3 vs E4; noise-rate curves; test if filtering deletes relevant passages; k sensitivity if time.
5. Efficiency: training/runtime on the actual machine; document model and hardware settings.

**Important:** At noise=0 or noise=.4/.8 in k=5 setup, every context contains >=1 labelled positive document. At noise=1 all contexts lack positive documents. Therefore balanced accuracy at a single noise stratum is not defined in the usual two-class sense; present conditional rates per stratum, balanced accuracy across combined conditions.

## Failure analysis template

| Question ID | Noise | Method | Positive passage present? | Predicted answer | Is answer supported by *shown* docs? | Error category |
|---|---:|---|---|---|---|---|
| | | | | | | |

Categories: unanswerable-but-answered, missed answerable, relevant doc filtered, contradictory evidence, generator used unstated prior knowledge, lexical-metric mismatch, truncation.

## Course deadline: 16 Nov 2026, 23:59 Singapore time

- **26–29 Oct:** Confirm topic/question; run split pipeline; verify RGB schema and train/val statistics.
- **30 Oct–3 Nov:** Train classifier; calibrate thresholds on validation; check all four conditions.
- **4–8 Nov:** Frozen-LM generation pilots, evaluation audits; finalize model choice and validation settings.
- **9–12 Nov:** Held-out final test; plots, error analysis, limitations; draft report.
- **13–15 Nov:** Team review, reproduce critical results, compile code ZIP + A4 PDF, verify AI Use Statement.
- **16 Nov:** Submit to NTULearn before 23:59.
