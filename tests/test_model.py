import unittest
import numpy as np
from rag_f.embedding import features
from rag_f.classifier import fit, probabilities, best_threshold, balanced_accuracy


class ModelTests(unittest.TestCase):
    def test_features(self):
        f = features(np.array([.2, .6, .4]), k=5)
        self.assertEqual(len(f), 8)
        self.assertAlmostEqual(f[0], .6, places=5)
        self.assertAlmostEqual(f[-1], .2, places=5)

    def test_train_classifier_small(self):
        # Artificial feature vectors for plumbing test; NOT RGB results.
        rng = np.random.default_rng(42)
        x = rng.normal(0, .05, size=(80, 8)).astype(np.float32)
        y = np.asarray([0]*40 + [1]*40, dtype=np.float32)
        x[40:, 0] += .4
        model, mean, std = fit(x[:60], y[:60], x[60:], y[60:], epochs=2)
        ps = probabilities(model, x[60:], mean, std)
        self.assertEqual(ps.shape, (20,))
        self.assertTrue(np.isfinite(ps).all())
        self.assertTrue(0 <= best_threshold(y[60:], ps) <= 1)

    def test_balanced_accuracy(self):
        y = np.array([0, 0, 1, 1])
        pred = np.array([0, 1, 1, 1])
        self.assertAlmostEqual(balanced_accuracy(y, pred), .75)


if __name__ == '__main__':
    unittest.main()
