"""Validation-only study of evidence filtering strategies for SC4001 Reliable RAG.

Run from project root:
    python scripts/compare_filters.py --split data/splits/val.jsonl --limit 30

This diagnostic uses benchmark passage labels; it does not estimate answer accuracy.
Choose a strategy on validation data and reserve independent test questions for final evaluation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_f.classifier import load
from rag_f.dataset import build_conditions, read_jsonl
from scripts.audit_filter import summarize


def filter_indices(scores, strategy: str, base_threshold: float):
    """Return kept passage indices without using ground-truth passage labels."""
    scores = [float(s) for s in scores]
    if not scores:
        return []
    if strategy == 'current':
        return [i for i, s in enumerate(scores) if s >= base_threshold]
    if strategy == 'relaxed_050':
        return [i for i, s in enumerate(scores) if s >= 0.50]
    if strategy == 'relaxed_045':
        return [i for i, s in enumerate(scores) if s >= 0.45]
    if strategy == 'top2':
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:2]
    if strategy == 'current_plus_top1':
        kept = [i for i, s in enumerate(scores) if s >= base_threshold]
        return kept if kept else [max(range(len(scores)), key=lambda i: scores[i])]
    raise ValueError(f'Unknown strategy: {strategy}')


def inspect_context(context: dict, scores, strategy: str, threshold: float) -> dict:
    """Evaluate label retention for a single synthesized retrieval context."""
    docs = context['documents']
    if len(scores) != len(docs):
        raise ValueError('Length of scores does not match documents')
    chosen = set(filter_indices(scores, strategy, threshold))
    pos_total = sum(d['source'] == 'positive' for d in docs)
    neg_total = sum(d['source'] == 'negative' for d in docs)
    pos_kept = sum(d['source'] == 'positive' for i, d in enumerate(docs) if i in chosen)
    neg_kept = sum(d['source'] == 'negative' for i, d in enumerate(docs) if i in chosen)
    return {
        'strategy': strategy,
        'id': context['id'],
        'noise': context['noise'],
        'before_positive': pos_total,
        'after_positive': pos_kept,
        'removed_positive': pos_total - pos_kept,
        'before_negative': neg_total,
        'after_negative': neg_kept,
        'removed_negative': neg_total - neg_kept,
        'lost_all_positive': int(pos_total > 0 and pos_kept == 0),
        'after_total': len(chosen),
        'duplicate_docs': context['duplicate_docs'],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='data/splits/val.jsonl')
    p.add_argument('--checkpoint', default='artifacts/evidence_head.pt')
    p.add_argument('--limit', type=int, default=30, help='Question count; use 0 for all validation questions')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='results/filter_strategies_validation.csv')
    a = p.parse_args()
    if 'test' in Path(a.split).name.lower():
        p.error('Use a validation split; do not select the filter strategy using test questions.')
    from rag_f.embedding import EvidenceEncoder

    _, _, _, config = load(a.checkpoint)
    rows = read_jsonl(a.split)
    if not rows:
        p.error('Validation split is empty')
    conditions = build_conditions(rows, config.get('noise_rates', [0.0, 0.4, 0.8, 1.0]),
                                  k=config['k'], seed=a.seed, limit=a.limit or None)
    encoder = EvidenceEncoder(config['encoder_name'])
    encoder.prime(conditions)
    scores = [encoder.similarities(c) for c in conditions]
    current = float(config['filter_threshold'])
    strategies = ('current', 'relaxed_050', 'relaxed_045', 'top2', 'current_plus_top1')

    all_rows = []
    for strat in strategies:
        all_rows.extend(inspect_context(context, sim, strat, current)
                        for context, sim in zip(conditions, scores))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'Validation questions: {len(rows) if a.limit == 0 else min(len(rows), a.limit)}; '
          f'contexts: {len(conditions)}; checkpoint threshold: {current:.3f}')
    print('Strategy            noise pos_kept neg_removed lost_all_pos avg_docs')
    for strat in strategies:
        subset = [r for r in all_rows if r['strategy'] == strat]
        for s in summarize(subset):
            pk = f"{s['positive_passage_retention']:.1%}" if s['positive_passage_retention'] is not None else 'N/A'
            nr = f"{s['negative_passage_removal']:.1%}" if s['negative_passage_removal'] is not None else 'N/A'
            la = f"{s['contexts_losing_all_positive']}/{s['answerable_contexts']}" if s['answerable_contexts'] else 'N/A'
            print(f"{strat:<20}{s['noise']:>4.1f} {pk:>8} {nr:>11} {la:>12} {s['mean_docs_after']:>8.2f}")
    print(f'Saved: {out}')
    print('Passage labels are proxies; these results do NOT measure final RAG answer quality.')


if __name__ == '__main__':
    main()
