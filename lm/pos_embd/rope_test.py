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

        # Check features are interleaved correctly: recall RoPE operates on pairs, so:
        # `cos` and `sin` are of the form:
        # [cos(θ_1), cos(θ_1), cos(2θ_2), cos(2θ_2), ..., cos((D/2)θ_{D/2})], and
        # [sin(θ_1), sin(θ_1), sin(2θ_2), sin(2θ_2), ..., sin((D/2)θ_{D/2})], resp.
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

    def test_forward(self):
        # Simple case: a single sequence of 3 tokens with a 2D embedding space
        B, H, T, D = 1, 1, 3, 2

        # For simplicity of testing, we ensure that all inverse frequencies are 1,
        # regardless of the feature index. This way, the frequencies only depend on
        # the token position.
        self.rope = RoPE(rope_theta=1, head_dim=D)

        q = torch.randn((B, H, T, D), dtype=torch.float32)
        k = torch.randn((B, H, T, D), dtype=torch.float32)
        q_rope, k_rope = self.rope(q, k)

        # Verify that the query (and key) before and after RoPE are different
        assert not torch.equal(q, q_rope)
        assert not torch.equal(k, k_rope)

        # Check that RoPE preserves the length of the embeddings because it simply
        # rotates the embeddings
        assert torch.allclose(torch.norm(q, dim=-1), torch.norm(q_rope, dim=-1))
        assert torch.allclose(torch.norm(k, dim=-1), torch.norm(k_rope, dim=-1))

        # Check position 0: RoPE should be the identity rotation in this case
        # since mθ = 0 for position m=0 regardless of the value of θ
        assert torch.equal(q_rope[..., 0, :], q[..., 0, :])
        assert torch.equal(k_rope[..., 0, :], k[..., 0, :])

        # Check position 1
        cos = torch.cos(torch.tensor(1.0))  # We ensured all inverse frequencies are 1
        sin = torch.sin(torch.tensor(1.0))  # We ensured all inverse frequencies are 1
        q1 = q[..., 1, 0]
        q2 = q[..., 1, 1]
        k1 = k[..., 1, 0]
        k2 = k[..., 1, 1]

        expected_q_rope_1 = q1 * cos - q2 * sin
        expected_q_rope_2 = q1 * sin + q2 * cos
        assert torch.equal(q_rope[..., 1, 0], expected_q_rope_1)
        assert torch.equal(q_rope[..., 1, 1], expected_q_rope_2)

        expected_k_rope_1 = k1 * cos - k2 * sin
        expected_k_rope_2 = k1 * sin + k2 * cos
        assert torch.equal(k_rope[..., 1, 0], expected_k_rope_1)
        assert torch.equal(k_rope[..., 1, 1], expected_k_rope_2)

        # Check position 2
        cos = torch.cos(torch.tensor(2.0))  # We ensured all inverse frequencies are 1
        sin = torch.sin(torch.tensor(2.0))  # We ensured all inverse frequencies are 1
        q1 = q[..., 2, 0]
        q2 = q[..., 2, 1]
        k1 = k[..., 2, 0]
        k2 = k[..., 2, 1]

        expected_q_rope_1 = q1 * cos - q2 * sin
        expected_q_rope_2 = q1 * sin + q2 * cos
        assert torch.equal(q_rope[..., 2, 0], expected_q_rope_1)
        assert torch.equal(q_rope[..., 2, 1], expected_q_rope_2)

        expected_k_rope_1 = k1 * cos - k2 * sin
        expected_k_rope_2 = k1 * sin + k2 * cos
        assert torch.equal(k_rope[..., 2, 0], expected_k_rope_1)
        assert torch.equal(k_rope[..., 2, 1], expected_k_rope_2)

        # Confirm that RoPE preserves the angle between the query and key
        cos_sim_before = torch.cosine_similarity(q, k, dim=-1)
        cos_sim_after = torch.cosine_similarity(q_rope, k_rope, dim=-1)
        assert torch.allclose(cos_sim_before, cos_sim_after)


if __name__ == "__main__":
    unittest.main()
