import unittest

import torch

from lm.pos_embd.rope import RoPE


class TestRoPE(unittest.TestCase):
    def setUp(self):
        self.B = 3
        self.T = 7
        self.D = 16
        self.rope = RoPE(rope_theta=2, head_dim=self.D)

    def tearDown(self):
        torch.cuda.empty_cache()

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
        B, T, D = 3, 7, 16
        rope = RoPE(rope_theta=2, head_dim=D)

        pos_ids = torch.arange(T).expand(B, T)
        cos, sin = rope.get_cos_sin(pos_ids)

        # Check shapes
        self.assertEqual(cos.shape, (B, T, D))
        self.assertEqual(sin.shape, (B, T, D))

        # Check independence across batches
        cos_ref = cos[0]
        sin_ref = sin[0]
        for b in range(1, B):
            assert torch.equal(cos[b], cos_ref)
            assert torch.equal(sin[b], sin_ref)

        # Check features are interleaved correctly (recall RoPE operates on pairs, so
        # `cos` is of the form [cos(θ), cos(θ), cos(2θ), cos(2θ), ..., cos((D/2)θ)]
        # `sin` is of the form [sin(θ), sin(θ), sin(2θ), sin(2θ), ..., sin((D/2)θ)]
        assert torch.equal(cos[..., 0::2], cos[..., 1::2])
        assert torch.equal(sin[..., 0::2], sin[..., 1::2])

    def test_swap_negate_pairwise(self):
        x = torch.tensor([[1, 2, 3, 4, 5, 6]])
        out = self.rope.swap_negate_pairwise(x)
        expected_out = torch.tensor([[-2, 1, -4, 3, -6, 5]])
        assert torch.equal(out, expected_out)

    def test_forward_identity_rotation(self):
        B, H, T, D = 1, 1, 1, 2
        rope = RoPE(rope_theta=2, head_dim=D)
        q = torch.randn((B, H, T, D), dtype=torch.float32)
        k = torch.randn((B, H, T, D), dtype=torch.float32)

        q_rope, k_rope = rope(q, k)

        # Confirm that RoPE is the identity rotation when T=1 (since there is only
        # position 0)
        assert torch.equal(q_rope, q)
        assert torch.equal(k_rope, k)

    def test_forward_one_rotation(self):
        B, H, T, D = 1, 1, 2, 2
        theta = 1
        self.rope = RoPE(rope_theta=theta, head_dim=D)

        q = torch.randn((B, H, T, D), dtype=torch.float32)
        k = torch.randn((B, H, T, D), dtype=torch.float32)
        q_rope, k_rope = self.rope(q, k)

        # Check position 0
        assert torch.equal(q_rope[..., 0, :], q[..., 0, :])
        assert torch.equal(k_rope[..., 0, :], k[..., 0, :])

        # Check position 1
        cos = torch.cos(1.0 * torch.tensor(theta))
        sin = torch.sin(1.0 * torch.tensor(theta))
        q1 = q[..., 1, 0]
        q2 = q[..., 1, 1]

        expected_q_rope_1 = q1 * cos - q2 * sin
        expected_q_rope_2 = q1 * sin + q2 * cos
        assert torch.equal(q_rope[..., 1, 0], expected_q_rope_1)
        assert torch.equal(q_rope[..., 1, 1], expected_q_rope_2)


if __name__ == "__main__":
    unittest.main()
