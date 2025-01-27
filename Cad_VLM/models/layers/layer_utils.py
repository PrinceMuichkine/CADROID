from typing import Literal, Union, Optional
import torch
from torch import Tensor

def perform_aggregate(
    X: Tensor,
    Y: Tensor,
    type: Literal["sum", "mean", "max"]
) -> Tensor:
    """
    Args:
        X: Input tensor of shape (B,N,D)
        Y: Input tensor of shape (B,N,D)
        type: Aggregation type - one of "sum", "mean", "max"
    Returns:
        Aggregated tensor of shape (B,N,D)
    
    Raises:
        ValueError: If type is not one of "sum", "mean", "max"
    """
    if type == "sum":
        return X + Y
    elif type == "mean":
        return 0.5 * (X + Y)
    elif type == "max":
        return torch.maximum(X, Y)
    else:
        raise ValueError(f"Unknown aggregation type: {type}. Must be one of: sum, mean, max")
