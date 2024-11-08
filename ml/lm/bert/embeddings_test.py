import unittest

import torch
from torch import nn

from ml.lm.bert.config import BertConfig
from ml.lm.bert.embeddings import BertEmbeddings


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        # Use simple toy configuration for the embedding layer
        self.config = BertConfig(
            vocab_size=1000, n_embd=64, max_len=128, n_segments=2, p_drop_hidden=0.1
        )
        self.B = 8  # Batch size
        self.T = 32  # Sequence length

    def test_init(self):
        # Test initialization of the embedding layer with positional and segment embeddings
        embeddings = BertEmbeddings(self.config, use_pos_emb=True, use_seg_emb=True)
        self.assertIsInstance(embeddings, nn.Module)
        self.assertIsNotNone(embeddings.lookup_tok_emb)
        self.assertIsNotNone(embeddings.lookup_pos_emb)
        self.assertIsNotNone(embeddings.lookup_seg_emb)

    def test_forward_without_position_and_segment_embeddings(self):
        embeddings = BertEmbeddings(self.config, use_pos_emb=False, use_seg_emb=False)
        x = torch.randint(0, self.config.vocab_size, (self.B, self.T))
        segments = torch.zeros(self.B, self.T, dtype=torch.long)

        # Check output
        output = embeddings(x, segments)
        self.assertEqual(output.shape, (self.B, self.T, self.config.n_embd))

    def test_forward_with_position_embeddings(self):
        embeddings = BertEmbeddings(self.config, use_pos_emb=True, use_seg_emb=False)
        x = torch.randint(0, self.config.vocab_size, (self.B, self.T))
        segment = torch.zeros(8, 32, dtype=torch.long)

        output = embeddings(x, segment)
        self.assertEqual(output.shape, (8, 32, self.config.n_embd))

    def test_forward_with_segment_embeddings(self):
        embeddings = BertEmbeddings(self.config, use_pos_emb=False, use_seg_emb=True)
        x = torch.randint(0, self.config.vocab_size, (self.B, self.T))
        segment = torch.randint(0, self.config.n_segments, (self.B, self.T))

        output = embeddings(x, segment)
        self.assertEqual(output.shape, (self.B, self.T, self.config.n_embd))

    def test_forward_with_position_and_segment_embeddings(self):
        embeddings = BertEmbeddings(self.config, use_pos_emb=True, use_seg_emb=True)
        x = torch.randint(0, self.config.vocab_size, (self.B, self.T))
        segment = torch.randint(0, self.config.n_segments, (self.B, self.T))

        output = embeddings(x, segment)
        self.assertEqual(output.shape, (self.B, self.T, self.config.n_embd))


if __name__ == "__main__":
    unittest.main()
