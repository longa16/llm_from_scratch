import torch
import torch.nn.functional as F
import math 

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """ 
    d_k = K.size(-1)
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    Weight = F.softmax(S, dim=-1)
    Output = torch.matmul(Weight, V)
    return Output