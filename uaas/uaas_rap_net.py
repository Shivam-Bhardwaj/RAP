"""
Uncertainty-Aware Adversarial Synthesis RAPNet model.
"""
import torch.nn as nn
from models.apr.rapnet import RAPNet

class UAASRAPNet(RAPNet):
    def __init__(self, args):
        super().__init__(args)
        # Get feature_dim from parent class after initialization
        # RAPNet uses hidden_dim from args for transformer output
        feature_dim = getattr(self, 'feature_dim', args.hidden_dim if hasattr(args, 'hidden_dim') else 256)
        # Add a head to predict aleatoric uncertainty (log variance)
        self.log_var_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6) # For 6-DoF pose (3 translation, 3 rotation as axis-angle)
        )

    def forward(self, x, return_feature=False):
        features, pose = super().forward(x, return_feature=True)
        log_variance = self.log_var_head(features)
        
        if return_feature:
            return (features, pose, log_variance)
        else:
            return (pose, log_variance)
