import math
import torch
import numpy as np
import copy
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional
from inspect import isfunction
from torch import nn, einsum
from einops import rearrange, repeat

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


def exists(val):
    return val is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, num_copies):
    return nn.ModuleList([copy.deepcopy(module) for i in range(num_copies)])

class GazeTransformer(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        x,
        context=None,
        attn_mask=None,
        query_pos_embed=None,
        context_pos_embed=None,
    ):
        output = x

        for layer in self.layers:
            output = layer(
                output,
                context=context,
                attn_mask=attn_mask,
                query_pos_embed=query_pos_embed,
                context_pos_embed=context_pos_embed
            )
        if type(output) == tuple:
            output, attn_weight, attn_weight_after = output
        
        if self.norm is not None:
            output = self.norm(output)

        return output




class CrossAttention_Layer(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.1, activation='relu'):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.linear1 = nn.Linear(query_dim, inner_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(inner_dim, query_dim)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
    
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, x, context=None, attn_mask=None, query_pos_embed=None, context_pos_embed=None):
        h = self.heads

        x_pos = self.with_pos_embed(x, query_pos_embed)
        context_pos = self.with_pos_embed(context, context_pos_embed)
        q = self.to_q(x_pos)
        k = self.to_k(context_pos)
        v = self.to_v(context_pos)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h).contiguous(), (q, k, v))
        attn_mask_repeat = attn_mask.unsqueeze(1).repeat(1, h, 1, 1).flatten(0, 1) if attn_mask is not None else None
        tgt2 = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                attn_mask=attn_mask_repeat
            )  # NOTE: attn_bias is added here

        tgt2 = rearrange(tgt2, '(b h) n d -> b n (h d)', h=h).contiguous()
        
        out = x + self.dropout2(tgt2)
        out = self.norm2(out)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = out + self.dropout3(tgt2)
        out = self.norm3(out)
        
        return out
    
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(query.size(0), L, S, dtype=query.dtype).to(query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = torch.bmm(query, key.transpose(-2, -1)) * scale_factor
        attn_weight = attn_weight.view(-1, L, S)
        #import pdb
        #pdb.set_trace()
        #attn_weight += attn_bias
        
        print('-------------------')
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        if scale is not None:
            attn_weight_after = attn_weight * scale
        
        
        return attn_weight @ value, attn_weight, attn_weight_after