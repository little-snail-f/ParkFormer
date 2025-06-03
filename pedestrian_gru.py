import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class PedestrianGru(nn.Module):
    def __init__(
        self,
        ped_input_dim=5,       
        hidden_dim=32,
        future_frames=4,
        embed_dim=256,
        teacher_forcing_ratio=0.5
    ):
        super().__init__()
        self.future_frames = future_frames
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # ========== (1) Encoder GRU ==========
        self.encoder_gru = nn.GRU(
            input_size=ped_input_dim,  
            hidden_size=hidden_dim,
            batch_first=True  
        )
        
    
        self.enc2dec_h = nn.Linear(hidden_dim, hidden_dim)
        
        # ========== (2) Decoder GRU ==========
        self.decoder_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # ========== (3) ==========
        self.real2embed = nn.Linear(3, embed_dim)

        # ========== (4) Output ==========
        # hidden_dim -> embed_dim
        self.hidden2embed = nn.Linear(hidden_dim, embed_dim)
        
        self.hidden2ped = nn.Linear(hidden_dim, 3)

    def forward(self, ped_motion, ped_future_gt):
        """
        Args:
            ped_motion:   [B,1,5] => (x, y, vx, vy, flag)
            ped_future_gt:[B,4,3] => (x, y, flag) (可选)
        Returns:
            ped_future_embed: [B,4,embed_dim]
            ped_future_pred:  [B,4,3]  => (x,y,flag)
        """
        B, seq_len, feat_dim = ped_motion.shape  # feat_dim=5
        device = ped_motion.device
        
        # ========== A. Encoder ==========
        enc_out, enc_h = self.encoder_gru(ped_motion)  
        
        enc_h = enc_h[-1]
        
        dec_h = self.enc2dec_h(enc_h).unsqueeze(0)
        
        # ========== B. Decoder ==========
        ped_future_embed = []
        ped_future_pred  = []
        
        dec_input = torch.zeros(B, 1, self.embed_dim, device=device)
        
        for t in range(self.future_frames):
            dec_out, dec_h = self.decoder_gru(dec_input, dec_h)  
            
            step_embed = self.hidden2embed(dec_out.squeeze(1)) 
            step_ped   = self.hidden2ped(dec_out.squeeze(1))    
            
            ped_future_embed.append(step_embed)
            ped_future_pred.append(step_ped)
            
            if (ped_future_gt is not None) and (random.random() < self.teacher_forcing_ratio):
                real_3d = ped_future_gt[:, t, :]
               
                real_embed = self.real2embed(real_3d)  

                dec_input = real_embed.unsqueeze(1)   
            else:
                dec_input = step_embed.unsqueeze(1)    

        
        ped_future_embed = torch.stack(ped_future_embed, dim=1)  
        ped_future_pred  = torch.stack(ped_future_pred,  dim=1)  
        
        return ped_future_embed, ped_future_pred