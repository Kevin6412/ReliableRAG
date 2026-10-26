import unittest
from rag_f.dataset import read_rgb, split_by_question, sample_context
from rag_f.generation import answer_match, is_abstention


def fake_rows(n=20):
    return [{'id': str(i), 'query': f'Who is person {i}?', 'answer': f'Name_{i}',
             'positive': [f'Person {i} is Name_{i}.', f'Person {i}: Name_{i}.'],
             'negative': [f'A fact about vegetable {i}.', 'Something irrelevant.']}
            for i in range(n)]


class DatasetTests(unittest.TestCase):
    def test_split_has_no_query_leakage(self):
        data = fake_rows(20)
        data.append({**data[0], 'id': 'duplicate-question'})
        splits = split_by_question(data, seed=42)
        question_sets = [{r['query'] for r in rows} for rows in splits.values()]
        self.assertEqual(len(set.union(*question_sets)), 20)
        for i in range(3):
            for j in range(i+1,3):
                self.assertFalse(question_sets[i] & question_sets[j])

    def test_sample_is_reproducible(self):
        row = fake_rows(1)[0]
        a = sample_context(row, noise=.6, k=5, seed=42)
        b = sample_context(row, noise=.6, k=5, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(sum(d['source'] == 'negative' for d in a['documents']), 3)
        self.assertEqual(a['answerable'], 1)
        self.assertEqual(sample_context(row, 1., k=5)['answerable'], 0)

    def test_no_gold_label_features_used_by_sample(self):
        r = sample_context(fake_rows(1)[0], 1., k=5)
        self.assertTrue(all(d['source'] == 'negative' for d in r['documents']))

    def test_answer_helpers(self):
        self.assertTrue(is_abstention('INSUFFICIENT_INFORMATION'))
        self.assertTrue(answer_match('The answer is Paris.', [['Paris', 'paris']]))
        self.assertFalse(answer_match('The answer is London.', 'Paris'))


if __name__ == '__main__':
    unittest.main()
