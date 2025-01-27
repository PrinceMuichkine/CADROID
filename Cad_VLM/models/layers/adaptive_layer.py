import torch
import copy
import torch.nn as nn
import os, sys
from typing import Dict, Any, Optional, Union, List, Tuple
from torch import Tensor

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))


from Cad_VLM.models.layers.attention import MultiHeadAttention
from Cad_VLM.models.layers.functional import FeedForwardLayer
from rich import print
from Cad_VLM.models.layers.utils_decode import generate_attention_mask
from Cad_VLM.models.utils import count_parameters


class AdaptiveLayer(nn.Module):
    """Adaptive attention layer for multi-modal feature fusion."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        
        # Attention layers
        self.self_attn = MultiHeadAttention(
            input_dim=input_dim,
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Feed forward layer
        self.feed_forward = FeedForwardLayer(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Device
        self.device = device
        
        # Attention scores storage
        self.attention_scores: Dict[str, Dict[str, Tensor]] = {}
        
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        metadata: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Any]]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            metadata: Whether to return attention metadata
        Returns:
            Tuple of:
            - Output tensor of shape (batch_size, seq_len, input_dim)
            - Optional metadata dictionary containing attention scores
        """
        # Self attention
        x2 = self.norm1(x)
        x2, self_attn_weights = self.self_attn(
            x2, x2, x2,
            need_weights=metadata,
            attn_mask=mask,
        )
        x = x + self.dropout(x2)
        
        # Feed forward
        x2 = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x2))
        
        # Store attention scores if needed
        if metadata and self_attn_weights is not None:
            self.attention_scores["self_attention"] = {"weights": self_attn_weights}
            return x, {"attention_scores": self.attention_scores}
            
        return x, None

    def total_parameters(self, description: bool = False, in_millions: bool = False) -> None:
        num_params = count_parameters(self, description)
        if in_millions:
            num_params_million = num_params / 1_000_000  # Convert to millions
            print(f"Number of Parameters: {num_params_million:.1f}M")
        else:
            num_params = count_parameters(self, description)
            print(f"Number of Parameters: {num_params}")

    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'AdaptiveLayer':
        return AdaptiveLayer(**config)


if __name__ == "__main__":
    adaptive_layer = AdaptiveLayer(4096).cuda()
    input_tensor = torch.rand(32, 512, 4096).cuda()

    attn_mask = generate_attention_mask(512, 512)

    output, attn_weight = adaptive_layer(
        input_tensor,
        {
            "attn_mask": None,
            "key_padding_mask": torch.randint(0, 2, (32, 512)).bool().cuda(),
        },
        metadata=True,
    )
    print(output.shape)

    print(attn_weight)
    print(adaptive_layer.total_parameters())
