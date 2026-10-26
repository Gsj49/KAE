import unittest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/gradient_mse'))
from boundary import tokenize_fixed_prompt


class Tokenizer:
    text_to_ids = {'P\n': [10, 11], 'P\nA': [10, 11, 20], 'A': [20],
                   'P\n \nB': [10, 12, 21], ' \nB': [22, 21], '': []}
    id_to_text = {10: 'P', 11: '\n', 12: '\n \n', 20: 'A', 21: 'B', 22: ' \n'}

    def __call__(self, text, **kwargs):
        return {'input_ids': torch.tensor([self.text_to_ids[text]], dtype=torch.long)}

    def decode(self, ids, **kwargs):
        return ''.join(self.id_to_text[int(index)] for index in ids)


class BoundaryTest(unittest.TestCase):
    def test_stable_tokens_unchanged(self):
        values, repairs = tokenize_fixed_prompt(Tokenizer(), 'P\n', torch.tensor([10, 11]), ['A'])
        self.assertEqual(values[0].tolist(), [20])
        self.assertEqual(repairs, [])

    def test_cross_boundary_merge_does_not_duplicate_prompt_suffix(self):
        tokenizer = Tokenizer()
        prompt = torch.tensor([10, 11])
        values, repairs = tokenize_fixed_prompt(tokenizer, 'P\n', prompt, ['A', ' \nB'])
        self.assertEqual(values[1].tolist(), [22, 21])
        self.assertEqual(tokenizer.decode(torch.cat([prompt, values[1]])), 'P\n \nB')
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0]['shared_prefix_tokens'], 1)
        self.assertTrue(torch.equal(prompt, torch.tensor([10, 11])))

    def test_empty_response_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Empty continuation'):
            tokenize_fixed_prompt(Tokenizer(), 'P\n', torch.tensor([10, 11]), [''])

    def test_lossy_response_rejected(self):
        class BadTokenizer(Tokenizer):
            def decode(self, ids, **kwargs):
                return super().decode(ids, **kwargs).replace('B', 'C')
        with self.assertRaisesRegex(ValueError, 'Response decoding'):
            tokenize_fixed_prompt(BadTokenizer(), 'P\n', torch.tensor([10, 11]), [' \nB'])


if __name__ == '__main__':
    unittest.main()
