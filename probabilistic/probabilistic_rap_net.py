"""
Probabilistic RAPNet using a Mixture Density Network head.
"""
import torch
import torch.nn as nn
import torch.distributions as D
from models.apr.rapnet import RAPNet

class ProbabilisticRAPNet(RAPNet):
    def __init__(self, args, num_gaussians=5):
        super().__init__(args)
        self.num_gaussians = num_gaussians
        # Get feature_dim from decoder_dim (transformer output dimension)
        decoder_dim = self.transformer_t.d_model
        feature_dim = getattr(self, 'feature_dim', decoder_dim)
        self.feature_dim = feature_dim * 2  # *2 for concatenated translation and rotation features
        self.mdn_head = nn.Linear(self.feature_dim, num_gaussians * (1 + 6 + 6)) # pi, mu, sigma for 6-DoF pose

    def forward(self, x, return_feature=False):
        # Call parent forward to get pose prediction
        pose_output = super().forward(x, return_feature=return_feature)
        
        # Extract features from parent - need to call with return_feature=True
        B, C, H, W = x.shape
        feature_maps, (pos_t, pos_rot) = self.backbone(x)
        features_t, features_rot = feature_maps
        
        # Get global descriptors from transformers
        pose_token_embed_rot = self.pose_token_embed_rot.unsqueeze(1).expand(-1, B, -1)
        pose_token_embed_t = self.pose_token_embed_t.unsqueeze(1).expand(-1, B, -1)
        local_descs_t = self.transformer_t(self.input_proj_t(features_t), pos_t, pose_token_embed_t)
        local_descs_rot = self.transformer_rot(self.input_proj_rot(features_rot), pos_rot, pose_token_embed_rot)
        global_desc_t = local_descs_t[:, 0, :]  # [B, decoder_dim]
        global_desc_rot = local_descs_rot[:, 0, :]  # [B, decoder_dim]
        
        # Concatenate for MDN input
        features = torch.cat([global_desc_t, global_desc_rot], dim=1)  # [B, decoder_dim*2]
        
        mdn_params = self.mdn_head(features)
        
        pi_logits, mu, log_sigma = mdn_params.split([self.num_gaussians, 
                                                     self.num_gaussians * 6, 
                                                     self.num_gaussians * 6], dim=-1)
        
        pi = D.Categorical(logits=pi_logits)
        mus = mu.view(-1, self.num_gaussians, 6)
        # Clamp log_sigma for numerical stability
        log_sigma_clamped = torch.clamp(log_sigma, min=-10, max=10)
        sigmas = torch.exp(log_sigma_clamped).view(-1, self.num_gaussians, 6)
        # Ensure minimum sigma to avoid numerical issues
        sigmas = torch.clamp(sigmas, min=1e-6)
        component_distribution = D.Independent(D.Normal(loc=mus, scale=sigmas), 1)
        mixture_distribution = D.MixtureSameFamily(pi, component_distribution)

        if return_feature:
            return (features, mixture_distribution)
        else:
            return mixture_distribution
