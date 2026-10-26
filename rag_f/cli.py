"""Command line entry: prepare, train, evaluate."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
from .dataset import read_rgb, read_jsonl, split_by_question, write_jsonl, build_conditions


def parse_rates(s: str) -> list[float]:
    values = [float(v) for v in s.split(',')]
    if not values or any(not 0. <= n <= 1. for n in values):
        raise argparse.ArgumentTypeError('Noise rates must be comma-separated in [0,1]')
    return values


def prepare(args):
    rows = read_rgb(args.input)
    splits = split_by_question(rows, args.seed)
    for name, part in splits.items():
        target = Path(args.out) / f'{name}.jsonl'
        write_jsonl(target, part)
        print(f'{name}: {len(part)} examples -> {target}')
    metadata = {'seed': args.seed, 'source': str(args.input),
                'n_unique_queries': len({r['query'].casefold() for r in rows}),
                'split_sizes': {name: len(part) for name, part in splits.items()}}
    with open(Path(args.out) / 'split_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def train(args):
    from .embedding import EvidenceEncoder, featurize
    from .classifier import fit, probabilities, best_threshold, save
    train_c = build_conditions(read_jsonl(args.train), args.noise_rates,
                               k=args.k, seed=args.seed, limit=args.limit)
    val_c = build_conditions(read_jsonl(args.val), args.noise_rates,
                             k=args.k, seed=args.seed, limit=args.limit)
    encoder = EvidenceEncoder(args.encoder)
    print(f'Building frozen embeddings: train={len(train_c)}, val={len(val_c)} contexts')
    train_x, train_y = featurize(train_c, encoder, k=args.k)
    val_x, val_y = featurize(val_c, encoder, k=args.k)
    model, mean, std = fit(train_x, train_y, val_x, val_y,
                           epochs=args.epochs, lr=args.lr, seed=args.seed)
    scores = probabilities(model, val_x, mean, std)
    cutoff = best_threshold(val_y, scores)
    sim_cutoff = best_threshold(val_y, val_x[:,0], grid=np.linspace(-.2, .9, 45))
    # Tune document filtering on validation labels only; never touch test labels.
    val_doc_pairs = [(float(sim), doc['source'] == 'positive')
                     for c in val_c for sim, doc in zip(encoder.similarities(c), c['documents'])]
    doc_scores = np.asarray([v[0] for v in val_doc_pairs])
    doc_labels = np.asarray([v[1] for v in val_doc_pairs], dtype=bool)
    filter_candidates = np.linspace(-.2, .9, 45)
    def f1(th):
        chosen = doc_scores >= th
        tp = int(np.sum(chosen & doc_labels))
        fp = int(np.sum(chosen & ~doc_labels))
        fn = int(np.sum(~chosen & doc_labels))
        return 2*tp / max(1, 2*tp+fp+fn)
    filter_cutoff = max(filter_candidates, key=lambda th: (f1(th), -abs(th-.3)))
    save(args.out, model, mean, std, model_name=args.encoder, k=args.k,
         noise_rates=args.noise_rates, neural_threshold=cutoff,
         similarity_threshold=sim_cutoff, filter_threshold=filter_cutoff)
    print('Saved', args.out)
    print(f'Validation-only thresholds: neural={cutoff:.3f}, similarity={sim_cutoff:.3f}')
    print(f'Validation-only filter threshold: {filter_cutoff:.3f}')
    print('DO NOT select checkpoints/thresholds based on the test set.')


def evaluate(args):
    from .embedding import EvidenceEncoder
    from .classifier import load
    from .evaluation import evaluate as evaluate_models, write_results
    model, mean, std, config = load(args.checkpoint)
    contexts = build_conditions(read_jsonl(args.test), args.noise_rates,
                                k=config['k'], seed=args.seed, limit=args.limit)
    encoder = EvidenceEncoder(config['encoder_name'])
    predictions = evaluate_models(contexts, encoder, model, mean, std, config,
                                  generator_name=args.generator,
                                  filtering_threshold=(args.filter_threshold if args.filter_threshold is not None
                                                       else config['filter_threshold']))
    summary = write_results(args.out, predictions)
    print(f'Wrote {args.out}/predictions.jsonl and summary.json')
    for row in summary:
        if row['noise'] == 'all':
            print(row['method'], 'N=', row['n'],
                  'gate_balanced_acc=', f"{row['gate_balanced_accuracy']:.3f}",
                  'gate_coverage=', f"{row['gate_coverage']:.3f}")
    if args.generator is None:
        print('EVIDENCE-ONLY evaluation: generation and answer quality NOT measured.')


def main():
    parser = argparse.ArgumentParser(description='SC4001 Reliable RAG project')
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Split English RGB data by unique question')
    prep.add_argument('--input', default='data/en_refine.json')
    prep.add_argument('--out', default='data/splits')
    prep.add_argument('--seed', type=int, default=42)
    prep.set_defaults(function=prepare)

    tr = commands.add_parser('train', help='Train neural evidence sufficiency head')
    tr.add_argument('--train', default='data/splits/train.jsonl')
    tr.add_argument('--val', default='data/splits/val.jsonl')
    tr.add_argument('--out', default='artifacts/evidence_head.pt')
    tr.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    tr.add_argument('--noise-rates', type=parse_rates, default=[0., .4, .8, 1.])
    tr.add_argument('--epochs', type=int, default=40)
    tr.add_argument('--lr', type=float, default=.001)
    tr.add_argument('--k', type=int, default=5)
    tr.add_argument('--limit', type=int)
    tr.add_argument('--seed', type=int, default=42)
    tr.set_defaults(function=train)

    ev = commands.add_parser('evaluate', help='Evaluate four methods on held-out test questions')
    ev.add_argument('--test', default='data/splits/test.jsonl')
    ev.add_argument('--checkpoint', default='artifacts/evidence_head.pt')
    ev.add_argument('--noise-rates', type=parse_rates, default=[0., .4, .8, 1.])
    ev.add_argument('--generator', default=None, help='e.g. google/flan-t5-small; omit for evidence-only analysis')
    ev.add_argument('--filter-threshold', type=float, default=None,
                    help='Optional override; by default use validation-calibrated threshold')
    ev.add_argument('--seed', type=int, default=42)
    ev.add_argument('--limit', type=int, help='Pilot only. Omit for final experiment')
    ev.add_argument('--out', default='results/test')
    ev.set_defaults(function=evaluate)

    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
