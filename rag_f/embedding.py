"""Frozen neural text encoder; score only question and document text."""
import numpy as np


class EvidenceEncoder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.cache: dict[str, np.ndarray] = {}

    def prime(self, contexts: list[dict], batch_size: int = 32) -> None:
        items = set()
        for c in contexts:
            items.add(c['query'])
            items.update(d['text'] for d in c['documents'])
        missing = sorted(items - self.cache.keys())
        if not missing:
            return
        values = self.model.encode(missing, normalize_embeddings=True,
                                   batch_size=batch_size, show_progress_bar=True)
        self.cache.update({text: vec for text, vec in zip(missing, values)})

    def similarities(self, context: dict) -> np.ndarray:
        q = self.cache[context['query']]
        return np.asarray([float(np.dot(q, self.cache[d['text']]))
                           for d in context['documents']], dtype=np.float32)


def features(scores: np.ndarray, k: int = 5) -> np.ndarray:
    """Top-k cosine similarities plus mean, dispersion and top-score margin.

    Padding -1 means 'missing document'; no true positive/negative labels leak.
    """
    values = np.sort(np.asarray(scores, dtype=np.float32))[::-1]
    top = np.full(k, -1., dtype=np.float32)
    top[:min(k, len(values))] = values[:k]
    if len(values) == 0:
        return np.concatenate([top, [-1., 0., 0.]]).astype(np.float32)
    gap = float(values[0]-values[1]) if len(values) > 1 else 0.
    return np.concatenate([top, [values.mean(), values.std(), gap]]).astype(np.float32)


def featurize(contexts: list[dict], encoder: EvidenceEncoder, k: int = 5):
    encoder.prime(contexts)
    xs = np.stack([features(encoder.similarities(c), k=k) for c in contexts])
    ys = np.asarray([c['answerable'] for c in contexts], dtype=np.float32)
    return xs, ys
