"""
Uncertainty-Aware Adversarial Synthesis RAPNet model.
"""
import torch
import torch.nn as nn
from models.apr.rapnet import RAPNet
from utils.pose_utils import compute_rotation_matrix_from_ortho6d

class UAASRAPNet(RAPNet):
    def __init__(self, args):
        super().__init__(args)
        # Get feature_dim from decoder_dim (transformer output dimension)
        # RAPNet uses hidden_dim from args for transformer, which is decoder_dim
        decoder_dim = self.transformer_t.d_model
        feature_dim = getattr(self, 'feature_dim', decoder_dim)
        # Add a head to predict aleatoric uncertainty (log variance)
        self.log_var_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),  # *2 for translation and rotation features concatenated
            nn.ReLU(),
            nn.Linear(128, 6) # For 6-DoF pose (3 translation, 3 rotation as axis-angle)
        )
        self.feature_dim = feature_dim * 2  # Store for reference

    def forward(self, x, return_feature=False):
        # Get features from backbone before transformer
        B, C, H, W = x.shape
        feature_maps, (pos_t, pos_rot) = self.backbone(x)
        features_t, features_rot = feature_maps
        
        # Get global descriptors from transformers (same as parent class)
        pose_token_embed_rot = self.pose_token_embed_rot.unsqueeze(1).expand(-1, B, -1)
        pose_token_embed_t = self.pose_token_embed_t.unsqueeze(1).expand(-1, B, -1)
        local_descs_t = self.transformer_t(self.input_proj_t(features_t), pos_t, pose_token_embed_t)
        local_descs_rot = self.transformer_rot(self.input_proj_rot(features_rot), pos_rot, pose_token_embed_rot)
        global_desc_t = local_descs_t[:, 0, :]  # [B, decoder_dim]
        global_desc_rot = local_descs_rot[:, 0, :]  # [B, decoder_dim]
        
        # Concatenate translation and rotation features for uncertainty prediction
        combined_features = torch.cat([global_desc_t, global_desc_rot], dim=1)  # [B, decoder_dim*2]
        
        # Get pose prediction (same as parent)
        x_t = self.regressor_head_t(global_desc_t)
        x_rot = self.regressor_head_rot(global_desc_rot)
        x_rot = compute_rotation_matrix_from_ortho6d(x_rot)
        x_t = x_t.reshape(B, 3, 1)
        x_rot = x_rot.reshape(B, 3, 3)
        pose = torch.cat((x_rot, x_t), dim=2)
        predicted_pose = pose.reshape(B, 12)
        
        # Predict uncertainty
        log_variance = self.log_var_head(combined_features)
        
        if return_feature:
            return (combined_features, predicted_pose, log_variance)
        else:
            return (predicted_pose, log_variance)
