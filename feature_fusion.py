import torch

from torch import nn
from timm.models.layers import trunc_normal_
from tool.config import Configuration


class FeatureFusion(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg

        tf_layer = nn.TransformerEncoderLayer(d_model=320, nhead=self.cfg.tf_en_heads)
        self.tf_encoder = nn.TransformerEncoder(tf_layer, num_layers=self.cfg.tf_en_layers) 

        total_length = 256
        self.pos_embed = nn.Parameter(torch.randn(1, total_length, 320) * .02)
        self.pos_drop = nn.Dropout(self.cfg.tf_en_dropout) 

        self.init_weights()

    
    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, bev_tar_feature, ego_ped_feature):   
        fuse_feature = torch.cat([bev_tar_feature, ego_ped_feature], dim=2) 

        fuse_feature = self.pos_drop(fuse_feature + self.pos_embed)

        fuse_feature = fuse_feature.transpose(0, 1) 
        fuse_feature = self.tf_encoder(fuse_feature) 
        
        fuse_feature = fuse_feature.transpose(0, 1) 
        return fuse_feature


