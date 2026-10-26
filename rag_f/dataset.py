"""Load RGB benchmark data, split by question, and construct controlled contexts."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


def read_rgb(path: str | Path) -> list[dict]:
    """Read RGB en_refine.json, which is JSONL despite its .json extension.

    Also supports a conventional JSON array for synthetic fixtures/tests.
    """
    with open(path, encoding='utf-8-sig') as f:
        first_nonspace = ''
        while True:
            c = f.read(1)
            if not c or not c.isspace():
                first_nonspace = c
                break
        f.seek(0)
        if first_nonspace == '[':
            rows = json.load(f)
        else:
            rows = []
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f'Invalid JSONL record at line {line_no} in {path}: {exc}'
                    ) from exc
    if not isinstance(rows, list):
        raise ValueError('Expected RGB records as JSONL or a JSON list')
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        query = row.get('query')
        positive = row.get('positive')
        negative = row.get('negative')
        answer = row.get('answer')
        if not isinstance(query, str) or not query.strip():
            continue
        if not isinstance(positive, list) or not isinstance(negative, list):
            continue
        if not positive or not negative or answer is None:
            continue
        # This starter is intentionally only for en_refine; other RGB splits differ.
        positive = [p.strip() for p in positive if isinstance(p, str) and p.strip()]
        negative = [p.strip() for p in negative if isinstance(p, str) and p.strip()]
        if not positive or not negative:
            continue
        result.append({'id': str(row.get('id', index)), 'query': query.strip(),
                       'answer': answer, 'positive': positive, 'negative': negative})
    if not result:
        raise ValueError('No valid examples; verify the RGB English refined dataset/schema')
    return result


def split_by_question(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """Split unique question groups 60/20/20; no question leaks across splits."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row['query'].strip().casefold(), []).append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_test = max(1, round(n * .2))
    n_val = max(1, round(n * .2))
    if n - n_test - n_val < 1:
        raise ValueError('At least 3 distinct questions required')
    partitions = {'test': keys[:n_test], 'val': keys[n_test:n_test+n_val],
                  'train': keys[n_test+n_val:]}
    return {part: [row for key in keys_in_partition for row in groups[key]]
            for part, keys_in_partition in partitions.items()}


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_context(row: dict, noise: float, *, k: int = 5, seed: int = 42) -> dict:
    """Synthetic retrieval: choose (1-noise)*k positives and noise*k negatives.

    If a pool is small, repeat samples (with replacement); log duplicate count.
    This makes requested noise exact but does not simulate a real search index.
    """
    if not 0 <= noise <= 1:
        raise ValueError('noise must be between 0 and 1')
    if k < 1:
        raise ValueError('k must be >= 1')
    ident = f"{row['id']}|{row['query']}|{noise:.6f}|{seed}"
    local_seed = int(hashlib.sha256(ident.encode('utf-8')).hexdigest()[:16], 16)
    rng = random.Random(local_seed)
    n_neg = round(k * noise)
    n_pos = k - n_neg
    positives = row['positive']
    negatives = row['negative']

    def draw(pool: list[str], quantity: int) -> list[str]:
        if len(pool) >= quantity:
            return rng.sample(pool, quantity)
        picked = list(pool)
        picked.extend(rng.choice(pool) for _ in range(quantity-len(pool)))
        return picked

    documents = ([{'text': p, 'source': 'positive'} for p in draw(positives, n_pos)] +
                 [{'text': p, 'source': 'negative'} for p in draw(negatives, n_neg)])
    rng.shuffle(documents)
    return {'id': row['id'], 'query': row['query'], 'answer': row['answer'],
            'documents': documents, 'noise': noise,
            'answerable': int(n_pos > 0),
            'duplicate_docs': len(documents)-len(set(d['text'] for d in documents))}


def build_conditions(rows: list[dict], rates: list[float], *, k=5, seed=42,
                     limit: int | None = None) -> list[dict]:
    if limit is not None:
        rows = rows[:limit]
    return [sample_context(row, rate, k=k, seed=seed)
            for row in rows for rate in rates]
