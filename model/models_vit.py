# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from multiview.epipolar import Epipolar_Attn
from einops import rearrange

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x, spatial_only=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if spatial_only:
            outcome =  x[:, 1:, :]
            return outcome
        
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    
    
class VisionTransformer_EpiAtt(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, sample_num=64, sim_type='softmax', **kwargs):
        super(VisionTransformer_EpiAtt, self).__init__(**kwargs)
        print("Sim type: ", sim_type)
        print("sample num: ", sample_num) 
        self.feat_size = (32, 24)    # 512//16, 384//16
        #self.feat_size = (14, 14)
        self.epi_attn_layers = nn.ModuleList()
        epi_attn_blk_1 = Epipolar_Attn(self.feat_size, 768, sample_size=sample_num, downsample_x=16,
                                             downsample_y=16, attn_heads=4, dim_head=64, sim_type=sim_type)
        self.epi_attn_layers.append(epi_attn_blk_1)
        
        epi_attn_blk_3 = Epipolar_Attn(self.feat_size, 768, sample_size=sample_num, downsample_x=16,
                                             downsample_y=16, attn_heads=4, dim_head=64, sim_type=sim_type)
        self.epi_attn_layers.append(epi_attn_blk_3)
        self.feature_w, self.feature_h = self.feat_size
        
        self.epipolar_steps=[6, 12]
         
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        
        print("Total transformer blocks: ", len(self.blocks))    
        
    def forward_features(self, x, spatial_only=False, fund_mat_v1=None, fund_mat_v2=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        blk_step = len(self.blocks) // len(self.epi_attn_layers)      

        attn_all, loc_mask_all = [], []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
                
            if (idx + 1) in self.epipolar_steps:
                
                attn_idx = (idx + 1) // blk_step - 1
                x_spatial = x[:, 1:, :]
                x_spatial = rearrange(x_spatial, 'b (h w) n -> b n h w', h=self.feature_h, w=self.feature_w).contiguous()
                c = x_spatial.size(1)
                x_spatial = x_spatial.view(-1, 2, c, self.feature_h, self.feature_w)
                x_v1, x_v2 = x_spatial[:, 0], x_spatial[:, 1]
                x_v1_new = self.epi_attn_layers[attn_idx](x_v1, x_v2, fund_mat_v1)
                
                x_v2_new = self.epi_attn_layers[attn_idx](x_v2, x_v1, fund_mat_v2)    
                
                if type(x_v2_new)==tuple:
                    x_v2_new, attn_weights, loc_valid_mask = x_v2_new
                
                x_spatial = torch.stack([x_v1_new, x_v2_new], dim=1).view(B, c, self.feature_h, self.feature_w)
                x_spatial = rearrange(x_spatial, 'b n h w -> b (h w) n', h=self.feature_h, w=self.feature_w).contiguous()
                x = torch.cat([x[:, 0:1], x_spatial.view(B, -1, c)], dim=1)
        
        #attn_all, loc_mask_all = torch.stack(attn_all, dim=1), torch.stack(loc_mask_all, dim=1)        
        
        if spatial_only:
            outcome =  x[:, 1:, :]
            return outcome, attn_all, loc_mask_all  
        
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        
        return outcome
    

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16_epipolar(**kwargs):
    model = VisionTransformer_EpiAtt(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model