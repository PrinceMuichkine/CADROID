from __future__ import division
from typing import Optional, Tuple, Union, List
from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Args:
        query, key, value: Input tensors
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        in_proj_weight: Input projection weight matrix
        in_proj_bias: Input projection bias vector
        bias_k, bias_v: Bias for keys and values
        add_zero_attn: Add a new batch of zeros to key and value sequences
        dropout_p: Dropout probability
        out_proj_weight: Output projection weight matrix
        out_proj_bias: Output projection bias vector
        training: Apply dropout if True
        key_padding_mask: Mask out padded keys
        need_weights: Return attention weights if True
        attn_mask: 2D or 3D mask preventing attention to certain positions
        use_separate_proj_weight: Use separate projection weights
        q_proj_weight, k_proj_weight, v_proj_weight: Separate projection weights
    Returns:
        Tuple of:
        - Output tensor of shape (batch_size, n_samples, embed_dim)
        - Optional attention weights tensor
    """
    # Get sizes
    tgt_len, bsz, embed_dim_to_check = query.size()
    assert embed_dim_to_check == embed_dim
    scaling = float(embed_dim) ** -0.5

    if use_separate_proj_weight:
        # Apply separate projections for q,k,v
        q = F.linear(query, q_proj_weight) if q_proj_weight is not None else query
        k = F.linear(key, k_proj_weight) if k_proj_weight is not None else key
        v = F.linear(value, v_proj_weight) if v_proj_weight is not None else value
    else:
        # Apply single projection for q,k,v
        if in_proj_weight is not None:
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        else:
            q = query
            k = key 
            v = value

    # Scale query
    q = q * scaling

    # Add bias
    if bias_k is not None:
        assert bias_v is not None
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # Add zero attention
    if add_zero_attn:
        zero_attn_shape = (k.shape[0], 1, k.shape[2])
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # Reshape q,k,v for multi-head attention
    q = q.contiguous().view(tgt_len, bsz * num_heads, -1).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, k.shape[-1] // num_heads).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, v.shape[-1] // num_heads).transpose(0, 1)

    # Calculate attention scores
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))

    # Apply attention mask
    if attn_mask is not None:
        attn_output_weights += attn_mask

    # Apply key padding mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, -1)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, -1)

    # Apply softmax and dropout
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    # Calculate attention output
    attn_output = torch.bmm(attn_output_weights, v)

    # Reshape output
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # Average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, -1)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class FeedForwardLayer(nn.Module):
    """Feed Forward Layer with residual connection."""
    
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim * 4)
        self.linear2 = nn.Linear(input_dim * 4, input_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, input_dim)
        """
        return self.linear2(self.activation(self.linear1(x)))

