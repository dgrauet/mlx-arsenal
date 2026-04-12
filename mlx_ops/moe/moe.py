import mlx.core as mx
import mlx.nn as nn


class MoEGate(nn.Module):
    """Top-k gating router for Mixture of Experts.

    Computes softmax scores over experts and returns top-k indices and weights.

    Reference: Hunyuan3D-2.1 moe_layers.py MoEGate

    Args:
        hidden_size: Input feature dimension.
        num_experts: Number of expert networks.
        top_k: Number of experts to route each token to.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Route tokens to top-k experts.

        Args:
            x: Input tensor of shape (num_tokens, hidden_size).

        Returns:
            indices: Expert indices of shape (num_tokens, top_k).
            weights: Routing weights of shape (num_tokens, top_k).
        """
        logits = self.gate(x)  # (num_tokens, num_experts)
        scores = mx.softmax(logits, axis=-1)
        indices = mx.argpartition(-scores, kth=self.top_k, axis=-1)[
            :, : self.top_k
        ]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        return indices, weights


class MoELayer(nn.Module):
    pass
