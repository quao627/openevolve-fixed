# EVOLVE-BLOCK-START
import torch
from torch import nn, einsum
import triton
import triton.language as tl


def custom_kernel(data):
    """
    Optimized TriMul (Triangle Multiplicative Update) kernel.

    Args:
        data: Tuple of (input_tensor, mask, weights, config)
            - input_tensor: [bs, seq_len, seq_len, dim]
            - mask: [bs, seq_len, seq_len]
            - weights: Dict[str, torch.Tensor] with keys:
                norm.weight, norm.bias, left_proj.weight, right_proj.weight,
                left_gate.weight, right_gate.weight, out_gate.weight,
                to_out_norm.weight, to_out_norm.bias, to_out.weight
            - config: Dict with dim, hidden_dim

    Returns:
        output: [bs, seq_len, seq_len, dim]
    """
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    batch_size, seq_len, _, _ = input_tensor.shape

    # Extract weights
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    left_proj_weight = weights["left_proj.weight"]
    right_proj_weight = weights["right_proj.weight"]
    left_gate_weight = weights["left_gate.weight"]
    right_gate_weight = weights["right_gate.weight"]
    out_gate_weight = weights["out_gate.weight"]
    to_out_norm_weight = weights["to_out_norm.weight"]
    to_out_norm_bias = weights["to_out_norm.bias"]
    to_out_weight = weights["to_out.weight"]

    # Step 1: LayerNorm
    x = input_tensor.to(torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + 1e-5)
    x = x * norm_weight + norm_bias

    # Step 2: Projections (linear layers without bias)
    left = x @ left_proj_weight.t()      # [bs, N, N, hidden_dim]
    right = x @ right_proj_weight.t()    # [bs, N, N, hidden_dim]

    # Step 3: Apply mask
    mask_expanded = mask.unsqueeze(-1)    # [bs, N, N, 1]
    left = left * mask_expanded
    right = right * mask_expanded

    # Step 4: Gating
    left_gate = (x @ left_gate_weight.t()).sigmoid()
    right_gate = (x @ right_gate_weight.t()).sigmoid()
    out_gate = (x @ out_gate_weight.t()).sigmoid()
    left = left * left_gate
    right = right * right_gate

    # Step 5: Einsum — the O(N^3) bottleneck
    # out[b,i,j,d] = sum_k left[b,i,k,d] * right[b,j,k,d]
    out = einsum('... i k d, ... j k d -> ... i j d',
                 left.to(torch.bfloat16), right.to(torch.bfloat16))
    out = out.to(torch.float32)

    # Step 6: Output LayerNorm
    out_mean = out.mean(dim=-1, keepdim=True)
    out_var = out.var(dim=-1, keepdim=True, unbiased=False)
    out = (out - out_mean) / torch.sqrt(out_var + 1e-5)
    out = out * to_out_norm_weight + to_out_norm_bias

    # Step 7: Output gating and projection
    out = out * out_gate
    out = out @ to_out_weight.t()

    return out
# EVOLVE-BLOCK-END
