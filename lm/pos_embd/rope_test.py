import unittest

import torch

from lm.pos_embd.rope import RoPE


class TestRoPE(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.T = 5
        self.D = 16
        self.rope = RoPE(rope_theta=2, head_dim=self.D)

    def test_get_cos_sin(self):
        pos_ids = torch.arange(self.T).expand(self.B, self.T)
        cos, sin = self.rope.get_cos_sin(pos_ids)
        for t in [cos, sin]:
            self.assertEqual(t.shape, (self.B, self.T, self.D))

    def test_rotate_half(self):
        x = torch.tensor([[1, 2, 3, 4, 5, 6]])
        assert torch.equal(
            self.rope.rotate_half(x), torch.tensor([[-4, -5, -6, 1, 2, 3]])
        )

        x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        assert torch.equal(
            self.rope.rotate_half(x), torch.tensor([[-5, -6, -7, -8, -9, 1, 2, 3, 4]])
        )

    def test_forward(self):
        B, T, H, D = 1, 2, 3, 4
        q = torch.randn((B, H, T, D), dtype=torch.float32)
        k = torch.randn((B, H, T, D), dtype=torch.float32)

        q_embd, k_embd = self.rope(q, k)

        assert q_embd.shape == q.shape
        assert k_embd.shape == k.shape


if __name__ == "__main__":
    unittest.main()
