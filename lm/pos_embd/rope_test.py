import unittest

import torch

from lm.pos_embd.rope import RoPE


class TestRoPE(unittest.TestCase):
    def setUp(self):
        self.B = 3
        self.T = 7
        self.D = 16
        self.rope = RoPE(rope_theta=2, head_dim=self.D)

    def test_init(self):
        self.assertEqual(
            self.rope.inv_freq.shape,
            (self.D // 2,),
            msg="The pre-computed inverse frequencies should have shape D/2 (where D "
            "is the head dimension) since RoPE operates on pairs of features.",
        )
        # Check the values of the inverse frequencies
        for i in range(0, self.D // 2):
            assert self.rope.inv_freq[i] == 1.0 / (2 ** (2 * i / self.D))

    def test_get_cos_sin(self):
        pos_ids = torch.arange(self.T).expand(self.B, self.T)
        cos, sin = self.rope.get_cos_sin(pos_ids)

        # Check shapes
        for t in [cos, sin]:
            self.assertEqual(t.shape, (self.B, self.T, self.D))

        # Check independence across batches
        cos_ref = cos[0]
        sin_ref = sin[0]
        for b in range(1, self.B):
            assert torch.equal(cos[b], cos_ref)
            assert torch.equal(sin[b], sin_ref)

        # Check that half of the features are duplicated (this is because the RoPE
        # operates on pairs of features)
        assert torch.equal(cos[..., : self.D // 2], cos[..., self.D // 2 :])
        assert torch.equal(sin[..., : self.D // 2], sin[..., self.D // 2 :])

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

    def test_forward_no_rotation(self):
        B, H, T, D = 1, 1, 1, 2
        self.rope = RoPE(rope_theta=2, head_dim=D)
        q = torch.randn((B, H, T, D), dtype=torch.float32)
        k = torch.randn((B, H, T, D), dtype=torch.float32)

        q_rope, k_rope = self.rope(q, k)

        # Confirm that RoPE is the identity rotation for D=2
        assert torch.equal(q_rope, q)
        assert torch.equal(k_rope, k)

    def test_forward_one_rotation(self):
        B, H, T, D = 1, 1, 2, 4
        theta = 2
        self.rope = RoPE(rope_theta=theta, head_dim=D)

        q = torch.randn((B, H, T, D), dtype=torch.float32)
        k = torch.randn((B, H, T, D), dtype=torch.float32)
        q_rope, k_rope = self.rope(q, k)

        # Check position 0
        assert torch.equal(q_rope[..., 0, :2], q[..., 0, :2])
        assert torch.equal(k_rope[..., 0, :2], k[..., 0, :2])

        # Check position 1
        theta = torch.tensor(theta, dtype=torch.float32)
        expected_q_rope_3 = (
            torch.cos(theta) * q[..., 1, :2] + torch.sin(theta) * q[..., 1, :2]
        )
        expected_q_rope_4 = (
            torch.sin(theta) * q[..., 1, :2] - torch.cos(theta) * q[..., 1, :2]
        )
        assert torch.allclose(q_rope[..., 1, :2], expected_q_rope_3)
        assert torch.allclose(q_rope[..., 1, :2], expected_q_rope_4)


if __name__ == "__main__":
    unittest.main()
