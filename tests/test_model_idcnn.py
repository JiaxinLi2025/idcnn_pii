import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_idcnn import IDCNNEncoder


class IDCNNEncoderTest(unittest.TestCase):
    def test_outputs_are_invariant_to_right_padding(self):
        torch.manual_seed(0)
        seq = torch.arange(1, 13).unsqueeze(0) % 49 + 1
        padded = torch.cat([seq, torch.zeros(1, 28, dtype=torch.long)], dim=1)

        for num_blocks in [1, 2, 3, 4]:
            with self.subTest(num_blocks=num_blocks):
                encoder = IDCNNEncoder(
                    vocab_size=50,
                    embedding_dim=8,
                    hidden_size=8,
                    dropout=0.0,
                    num_blocks=num_blocks,
                )
                encoder.eval()
                with torch.no_grad():
                    short_out = encoder(seq, seq.ne(0))
                    padded_out = encoder(padded, padded.ne(0))[:, : seq.size(1)]

                self.assertTrue(torch.allclose(short_out, padded_out, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
