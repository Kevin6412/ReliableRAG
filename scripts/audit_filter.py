"""Validation-only audit of the RGB document similarity filter.

Use from project root:
    python scripts/audit_filter.py --split data/splits/val.jsonl --limit 30

This audits *labelled passage retention*, not answer correctness or hallucination.
Do not tune filter thresholds on held-out test examples.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

# Running this file from scripts/ should still import rag_f at project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_f.dataset import build_conditions, read_jsonl


def analyze(context: dict, scores, threshold: float) -> dict:
    """Inspect which labelled positive/negative passages a filter removes."""
    documents = context['documents']
    if len(documents) != len(scores):
        raise ValueError('Scores and document count differ')
    kept = [doc for doc, score in zip(documents, scores) if float(score) >= threshold]
    before_positive = sum(d['source'] == 'positive' for d in documents)
    after_positive = sum(d['source'] == 'positive' for d in kept)
    before_negative = sum(d['source'] == 'negative' for d in documents)
    after_negative = sum(d['source'] == 'negative' for d in kept)
    return {
        'id': context['id'],
        'noise': context['noise'],
        'before_positive': before_positive,
        'after_positive': after_positive,
        'removed_positive': before_positive - after_positive,
        'before_negative': before_negative,
        'after_negative': after_negative,
        'removed_negative': before_negative - after_negative,
        'lost_all_positive': int(before_positive > 0 and after_positive == 0),
        'after_total': len(kept),
        'duplicate_docs': context['duplicate_docs'],
    }


def summarize(rows: list[dict]) -> list[dict]:
    by_noise = defaultdict(list)
    for r in rows:
        by_noise[r['noise']].append(r)
    output = []
    for noise, subset in sorted(by_noise.items()):
        pos_before = sum(r['before_positive'] for r in subset)
        pos_after = sum(r['after_positive'] for r in subset)
        neg_before = sum(r['before_negative'] for r in subset)
        neg_removed = sum(r['removed_negative'] for r in subset)
        n_with_pos = sum(r['before_positive'] > 0 for r in subset)
        n_lost_all = sum(r['lost_all_positive'] for r in subset)
        output.append({
            'noise': noise,
            'N': len(subset),
            'positive_passage_retention': pos_after/pos_before if pos_before else None,
            'negative_passage_removal': neg_removed/neg_before if neg_before else None,
            'contexts_losing_all_positive': n_lost_all,
            'answerable_contexts': n_with_pos,
            'mean_docs_after': sum(r['after_total'] for r in subset)/len(subset),
        })
    return output


def main():
    p = argparse.ArgumentParser(description='Audit labelled evidence removed by similarity filtering on validation')
    p.add_argument('--split', default='data/splits/val.jsonl', help='Validation data only; not held-out test')
    p.add_argument('--checkpoint', default='artifacts/evidence_head.pt')
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='results/filter_audit_validation.csv')
    args = p.parse_args()
    if Path(args.split).name.lower().startswith('test'):
        p.error('For method diagnosis, use val.jsonl, not test.jsonl')
    from rag_f.classifier import load
    from rag_f.embedding import EvidenceEncoder
    _, _, _, config = load(args.checkpoint)
    conditions = build_conditions(read_jsonl(args.split), config.get('noise_rates', [0., .4, .8, 1.]),
                                  k=config['k'], seed=args.seed, limit=args.limit)
    encoder = EvidenceEncoder(config['encoder_name'])
    encoder.prime(conditions)
    threshold = config['filter_threshold']
    rows = [analyze(c, encoder.similarities(c), threshold) for c in conditions]
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Validation-only filter cutoff: {threshold:.4f}; contexts: {len(rows)}')
    print('noise  N   pos_kept  neg_removed  lost_all_pos  avg_docs_after')
    for s in summarize(rows):
        pos = f"{s['positive_passage_retention']:.1%}" if s['positive_passage_retention'] is not None else 'N/A'
        neg = f"{s['negative_passage_removal']:.1%}" if s['negative_passage_removal'] is not None else 'N/A'
        lost = f"{s['contexts_losing_all_positive']}/{s['answerable_contexts']}" if s['answerable_contexts'] else 'N/A'
        print(f"{s['noise']:>4.1f} {s['N']:>3}  {pos:>8}  {neg:>11}  {lost:>12}  {s['mean_docs_after']:>14.2f}")
    print(f'Details: {destination}')
    print('NOTE: Positive/negative are RGB passage labels, not independent proof of factual support.')


if __name__ == '__main__':
    main()
