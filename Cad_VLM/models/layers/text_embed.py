from typing import Dict, Any, Optional, Union, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from ..utils import get_device, get_device_str


MODEL_NAME_DICT={"bert_large_uncased":"google-bert/bert-large-uncased"}

def prepare_cross_attention_mask_batch(mask, cad_seq_len=271):
    if mask.shape[0] > 1:
        length=mask.shape[1]
        batch_size=mask.shape[0]
        mask = mask.reshape(batch_size, 1, length)
    mask = torch.tile(mask, (1, cad_seq_len, 1))  # (512) -> (271, 512)
    mask = torch.where(
        mask, -torch.inf, 0
    )  # Changing the [True,False] format to [0,-inf] format

    return mask

class TextEmbedder(nn.Module):
    """Text embedding layer using pretrained transformer model."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Output projection if needed
        self.output_dim = output_dim
        if output_dim is not None:
            self.proj = nn.Linear(self.model.config.hidden_size, output_dim)
        
        # Config
        self.max_length = max_length
        
    def forward(self, texts: Union[str, List[str]]) -> Tensor:
        """
        Args:
            texts: Input text or list of texts
        Returns:
            Text embeddings tensor of shape:
            - (batch_size, seq_len, hidden_size) if output_dim is None
            - (batch_size, seq_len, output_dim) if output_dim is specified
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Get embeddings
        outputs = self.model(**tokens)
        embeddings = outputs.last_hidden_state
        
        # Project if needed
        if self.output_dim is not None:
            embeddings = self.proj(embeddings)
            
        return embeddings

    @staticmethod
    def from_config(config: dict):
        return TextEmbedder(
            **config
        )
        
