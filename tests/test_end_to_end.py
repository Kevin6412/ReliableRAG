"""Artificial integration test, with no pretrained model download."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rag_f.dataset import sample_context
from rag_f.embedding import features
from rag_f.classifier import make_model, save, load
from rag_f.evaluation import evaluate, summarize, write_results


class DummyEncoder:
    def prime(self, contexts):
        pass
    def similarities(self, context):
        return np.asarray([.71 if d['source'] == 'positive' else .13
                           for d in context['documents']], dtype=np.float32)


class TestPipeline(unittest.TestCase):
    def test_all_4_methods_write_metrics(self):
        samples = [{'id': i, 'query': f'Question {i}', 'answer': f'X{i}',
                    'positive': [f'X{i} is the answer'],
                    'negative': ['A different fact about another subject']}
                   for i in range(5)]
        contexts = [sample_context(row, noise, seed=42)
                    for row in samples for noise in (0., .4, 1.)]
        model = make_model(input_dim=len(features(np.asarray([.1]*5))))
        mean = np.zeros(8, dtype=np.float32)
        std = np.ones(8, dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / 'model.pt'
            save(file, model, mean, std,
                 model_name='STUB', k=5, noise_rates=[0., .4, 1.],
                 neural_threshold=.5, similarity_threshold=.4, filter_threshold=.3)
            loaded, m, sd, config = load(file)
            records = evaluate(contexts, DummyEncoder(), loaded, m, sd, config)
            self.assertEqual(len(records), len(contexts)*4)
            summary = write_results(Path(d) / 'results', records)
            self.assertTrue((Path(d)/'results'/'summary.json').exists())
            self.assertEqual(len([r for r in summary if r['noise'] == 'all']),4)
            self.assertTrue(all('lexical_accuracy_on_answerable' not in r for r in summary))
            self.assertTrue(all(r['gate_balanced_accuracy'] is None
                                for r in summary if r['noise'] != 'all'))

if __name__ == '__main__':
    unittest.main()
