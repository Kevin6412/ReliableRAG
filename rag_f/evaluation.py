"""Evidence-gating evaluation and optional answer generation, with cautious labels."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .classifier import probabilities, balanced_accuracy
from .generation import FrozenGenerator, INSUFFICIENT, answer_match, is_abstention


def evaluate(contexts, encoder, model, mean, std, config, *,
             generator_name=None, filtering_threshold=0.30):
    if generator_name:
        generator = FrozenGenerator(generator_name)
    else:
        generator = None
    encoder.prime(contexts)
    n_features = config['input_dim']
    from .embedding import features
    similarity_vectors = [encoder.similarities(c) for c in contexts]
    feature_matrix = np.stack([features(scores, k=config['k']) for scores in similarity_vectors])
    assert feature_matrix.shape[1] == n_features
    ps = probabilities(model, feature_matrix, mean, std)
    output = []
    for i, (context, sims) in enumerate(zip(contexts, similarity_vectors)):
        max_sim = float(sims.max()) if len(sims) else -1.
        selected = [doc['text'] for doc, score in zip(context['documents'], sims)
                    if score >= filtering_threshold]
        raw_docs = [doc['text'] for doc in context['documents']]
        for method in ['standard', 'similarity_gate', 'neural_gate', 'combined']:
            if method == 'standard':
                accepts = True
                docs = raw_docs
            elif method == 'similarity_gate':
                accepts = max_sim >= config['similarity_threshold']
                docs = raw_docs
            elif method == 'neural_gate':
                accepts = ps[i] >= config['neural_threshold']
                docs = raw_docs
            else:
                accepts = ps[i] >= config['neural_threshold'] and bool(selected)
                docs = selected
            answer = (generator.generate(context['query'], docs) if accepts
                      else INSUFFICIENT) if generator else None
            record = {'id': context['id'], 'noise': context['noise'],
                      'method': method, 'answerable_proxy': int(context['answerable']),
                      'accepts': bool(accepts), 'max_similarity': max_sim,
                      'neural_confidence': float(ps[i]), 'docs_before': len(raw_docs),
                      'docs_after': len(docs) if accepts else 0,
                      'duplicate_docs': context['duplicate_docs'],
                      'answer': answer,
                      'gold': context['answer']}
            if answer is not None:
                record['abstained'] = is_abstention(answer)
                record['lexical_answer_match'] = answer_match(answer, context['answer'])
            output.append(record)
    return output


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r['method'], str(r['noise']))].append(r)
        groups[(r['method'], 'all')].append(r)
    rows = []
    for (method, noise), sample in sorted(groups.items()):
        truth = np.asarray([r['answerable_proxy'] for r in sample], dtype=int)
        predicted = np.asarray([r['accepts'] for r in sample], dtype=bool)
        answerable = truth == 1
        unanswerable = ~answerable
        def conditional(arr, cond):
            return float(np.mean(np.asarray(arr)[cond])) if np.any(cond) else None
        row = {'method': method, 'noise': noise, 'n': len(sample),
               'answerable_n': int(answerable.sum()),
               'unanswerable_n': int(unanswerable.sum()),
               'gate_balanced_accuracy': (balanced_accuracy(truth, predicted)
                                          if np.any(answerable) and np.any(unanswerable)
                                          else None),
               'gate_coverage': float(np.mean(predicted)),
               'gate_answerable_recall': conditional(predicted, answerable),
               'gate_correct_abstain_rate': conditional(~predicted, unanswerable),
               'gate_false_answer_rate': conditional(predicted, unanswerable),
               'avg_duplicate_docs': float(np.mean([r['duplicate_docs'] for r in sample]))}
        if 'abstained' in sample[0]:
            abstained = [r['abstained'] for r in sample]
            matches = [r['lexical_answer_match'] for r in sample]
            row.update({'final_coverage': float(np.mean(~np.asarray(abstained))),
                        'lexical_accuracy_on_answerable': conditional(matches, answerable),
                        'final_unsupported_answer_rate_proxy':
                            conditional(~np.asarray(abstained), unanswerable),
                        'final_correct_abstain_rate': conditional(abstained, unanswerable)})
        rows.append(row)
    return rows


def write_results(path, predictions):
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(Path(path) / 'predictions.jsonl', 'w', encoding='utf-8') as f:
        for r in predictions:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    rows = summarize(predictions)
    with open(Path(path) / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return rows
