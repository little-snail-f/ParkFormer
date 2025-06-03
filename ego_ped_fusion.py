#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sin_cos_pos_encoding(seq_len, dim):
    """
    构建形如 [seq_len, dim] 的一维正余弦位置编码 (sine-cosine).
      pos: 序列位置 0,1,...,seq_len-1
      i:   通道维度 0,1,...,dim-1
    常见公式:
        PE(pos, 2i)   = sin(pos / (10000^(2i/dim)))
        PE(pos, 2i+1) = cos(pos / (10000^(2i/dim)))
    """
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len,1]
    i   = torch.arange(dim,   dtype=torch.float).unsqueeze(0)    # [1,dim]
    
    # div_term = 10000^(2*(i//2)/dim)，这里用 exp(-log(10000)*2*(i//2)/dim)
    div_term = torch.exp(- math.log(10000.0) * (2*(i//2)) / dim)
    
    # phase = pos * div_term => [seq_len,dim]
    phase = pos * div_term
    
    # 偶数下标用 sin，奇数下标用 cos
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(phase[:, 0::2])  # 偶数列
    pe[:, 1::2] = torch.cos(phase[:, 1::2])  # 奇数列
    
    return pe  # [seq_len, dim]


class CrossAttnBlock(nn.Module):
    """
    单层 Cross-Attention:
      - Q = Ego tokens
      - K,V = Ped tokens
      - 残差 + LayerNorm
      - FFN + 残差 + LayerNorm
    """
    def __init__(self, embed_dim=64, nhead=8, ff_dim=256, dropout=0.1):
        super().__init__()
        
        # 多头注意力: batch_first=True => 输入输出[B,N,E]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, ego_q, ped_kv):
        """
        Args:
          ego_q:  [B, Nq, E],   Nq=256,  E=64
          ped_kv: [B, Nk, E],   Nk=10
        Returns:
          out:    [B, Nq, E]   形状与Q相同
        """
        # ========== (1) Cross-Attention ==========
        attn_out, _ = self.cross_attn(query=ego_q, key=ped_kv, value=ped_kv)
        
        # 残差 + LayerNorm
        x = self.norm1(ego_q + attn_out)
        
        # ========== (2) FFN ==========
        ffn_out = self.ffn(x) 
        out = self.norm2(x + ffn_out)
        
        return out


class EgoPedTransformer(nn.Module):
    """
    将自车(ego)特征 [B,3] + 行人(行人轨迹预测) [B,4,256] 做融合:
      1) Ego => MLP => [B,256,64] + pos_encoding => ego_tokens
      2) Ped => linear => [B,4,64] + pos_encoding => ped_tokens
      3) 4层 CrossAttnBlock (Q=ego, K=ped, V=ped)
      4) 最终输出 [B,256,64]
    """
    def __init__(self, 
                 ego_in_dim=3,    # Ego motion 原始维度 (x,y,yaw) 或 (velocity,acc_x,acc_y)
                 ped_in_dim=256,  # 行人轨迹特征维度
                 seq_len=256, 
                 embed_dim=64,
                 nhead=8, 
                 ff_dim=256, 
                 dropout=0.1,
                 num_layers=4):
        super().__init__()
        
        self.seq_len   = seq_len
        self.embed_dim = embed_dim
        
        # --- (1) Ego motion => MLP => [B,seq_len*embed_dim] => [B,256,64] ---
        self.ego_mlp = nn.Sequential(
            nn.Linear(ego_in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, seq_len * embed_dim),  # => 256*64
            nn.ReLU(inplace=True),
        )
        
        # --- (2) Ped => [B,4,64] ---
        self.ped_proj = nn.Linear(ped_in_dim, embed_dim)
        
        # --- (3) 4层 CrossAttnBlock ---
        self.layers = nn.ModuleList([
            CrossAttnBlock(embed_dim=embed_dim, nhead=nhead, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # --- (4) 位置编码 (sine-cosine) ---
        # ego: 长度=256, ped: 长度=10
        pe_ego = build_sin_cos_pos_encoding(seq_len, embed_dim)  # [256,64]
        pe_ped = build_sin_cos_pos_encoding(4, embed_dim)       #  [4,64]

        # 不可训练 => register_buffer
        self.register_buffer("pos_ego", pe_ego.unsqueeze(0))  # => [1,256,64]
        self.register_buffer("pos_ped", pe_ped.unsqueeze(0))  # => [1,4,64]

    def forward(self, ego_motion, ped_future):
        """
        Args:
          ego_motion: [B,3]   自车运动 (例如 velocity, acc_x, acc_y)
          ped_future: [B,4,256] 行人轨迹预测
        Returns:
          out: [B,256,64]
        """
        B = ego_motion.shape[0]
        
        # ========== (A) Ego => [B,256,64] + pos_encoding ==========
        # 先 MLP => [B,256*64]，再 view => [B,256,64]
        ego_x = self.ego_mlp(ego_motion)
        ego_x = ego_x.view(B, self.seq_len, self.embed_dim)
        # 加正余弦位置编码
        ego_x = ego_x + self.pos_ego  # [1,256,64] => broadcast
        
        # ========== (B) Ped => [B,4,64] + pos_encoding ==========
        ped_x = self.ped_proj(ped_future)  # => [B,4,64]
        ped_x = ped_x + self.pos_ped       # => [B,4,64]
        
        # ========== (C) 4层 CrossAttnBlock ==========
        for layer in self.layers:
            ego_x = layer(ego_x, ped_x)
        
        return ego_x


