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
        if self.top_k >= self.num_experts:
            # All experts selected; just use argsort
            indices = mx.argsort(-scores, axis=-1)[:, : self.top_k]
        else:
            indices = mx.argpartition(-scores, kth=self.top_k, axis=-1)[
                :, : self.top_k
            ]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        return indices, weights


class MoELayer(nn.Module):
    """Mixture of Experts layer with optional shared expert.

    Routes tokens to top-k experts via a gating network, computes weighted
    sum of expert outputs, and optionally adds a shared expert's output.

    Reference: Hunyuan3D-2.1 moe_layers.py MoEBlock

    Args:
        hidden_size: Input feature dimension.
        num_experts: Number of routed expert networks.
        top_k: Number of experts per token.
        expert_fn: Callable that returns an nn.Module expert.
        shared_expert: Optional always-active expert module.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        expert_fn: callable,
        shared_expert: nn.Module | None = None,
    ):
        super().__init__()
        self.gate = MoEGate(hidden_size, num_experts, top_k)
        self.experts = [expert_fn() for _ in range(num_experts)]
        self.shared_expert = shared_expert

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with expert routing.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Output tensor of same shape as input.
        """
        orig_shape = x.shape
        hidden = x.reshape(-1, orig_shape[-1])  # (num_tokens, hidden_size)

        indices, weights = self.gate(hidden)  # (num_tokens, top_k) each

        # Compute per-expert weight: sum of routing weights where expert was selected
        # expert_weights: (num_tokens, num_experts)
        num_experts = len(self.experts)
        num_tokens = hidden.shape[0]
        expert_weights = mx.zeros((num_tokens, num_experts))
        for k in range(indices.shape[1]):
            col_indices = indices[:, k : k + 1]  # (num_tokens, 1)
            col_weights = weights[:, k : k + 1]  # (num_tokens, 1)
            # One-hot for this top-k slot
            one_hot = mx.zeros((num_tokens, num_experts))
            rows = mx.arange(num_tokens).reshape(-1, 1)
            one_hot[rows, col_indices] = col_weights
            expert_weights = expert_weights + one_hot

        # Run each expert on all tokens and accumulate weighted outputs
        output = mx.zeros_like(hidden)
        for expert_idx, expert in enumerate(self.experts):
            w = expert_weights[:, expert_idx : expert_idx + 1]  # (num_tokens, 1)
            if not mx.any(w > 0).item():
                continue
            expert_output = expert(hidden)  # (num_tokens, hidden_size)
            output = output + w * expert_output

        if self.shared_expert is not None:
            output = output + self.shared_expert(hidden)

        return output.reshape(orig_shape)
