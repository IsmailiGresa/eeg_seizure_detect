import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int, scale: bool = True, block_mask: list = None) -> None:
        super(ScaledDotProductAttention, self).__init__()
        if scale:
            self.sqrt_dim = np.sqrt(dim)
        else:
            self.sqrt_dim = 1
        self.block_mask = block_mask

        if self.block_mask is not None:
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, dim = -1)

        if self.block_mask is not None:
            attn = attn * self.block_mask
            context = self.gamma * torch.bmm(attn, value)
        else:
            context = torch.bmm(attn, value)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 8, block_mask: list = None) -> None:
        super(MultiHeadAttention, self).__init__()

        assert dim % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dim, self.d_head * num_heads)
        self.key_proj = nn.Linear(dim, self.d_head * num_heads)
        self.value_proj = nn.Linear(dim, self.d_head * num_heads)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head, scale = True, block_mask = block_mask)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)

        return context, attn