"""
Trainer for Multi-Hypothesis Probabilistic Absolute Pose Regression.
"""
from tqdm import tqdm
import numpy as np
import torch
from torch.cuda.amp import autocast

from rap import BaseTrainer
from probabilistic.probabilistic_rap_net import ProbabilisticRAPNet
from probabilistic.hypothesis_validator import HypothesisValidator
from probabilistic.loss import MixtureNLLLoss

class ProbabilisticTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        # Override model with Probabilistic version
        self.model = ProbabilisticRAPNet(args).to(args.device)
        self.hypothesis_validator = HypothesisValidator(self.renderer)
        self.criterion = MixtureNLLLoss()

    def train_epoch(self, epoch, poses_perturbed, imgs_perturbed):
        self.model.train()
        if self.args.freeze_batch_norm:
            self.model = self.freeze_bn_layer_train(self.model)

        train_loss_epoch = []

        selected_indexes = np.random.choice(self.dset_size, size=[self.dset_size], replace=False)

        i_batch = 0
        batch_size = self.args.batch_size
        device = self.args.device
        
        for _ in tqdm(range(self.n_iters), desc=f'Epoch {epoch}'):
            if i_batch + batch_size > self.dset_size:
                break
            batch_indexes = selected_indexes[i_batch:i_batch + batch_size]
            i_batch += batch_size

            imgs_normed_batch = self.imgs_normed[batch_indexes].to(device, non_blocking=True)
            poses_batch = self.poses[batch_indexes].reshape(batch_size, 12).to(device, torch.float, non_blocking=True)

            with autocast(device, enabled=self.args.amp, dtype=self.args.amp_dtype):
                mixture_distribution = self.model(imgs_normed_batch)
                loss = self.criterion(mixture_distribution, poses_batch)
                
                # Hypothesis validation (periodically for evaluation)
                if epoch % 50 == 0 and hasattr(self, 'val_dl'):
                    # Sample hypotheses and validate
                    n_hypotheses = 5
                    hypotheses = mixture_distribution.sample((n_hypotheses,))  # (n_hyp, batch, 6)
                    
                    # Validate hypotheses using rendering (for first sample in batch)
                    if batch_size > 0:
                        try:
                            # Get camera params for rendering
                            cam_params = getattr(self.renderer, 'cam_params', None)
                            if cam_params is not None:
                                observed_img = imgs_normed_batch[0]
                                hypotheses_batch = hypotheses[:, 0, :].reshape(n_hypotheses, 6)
                                
                                scores = self.hypothesis_validator.validate(
                                    hypotheses_batch, observed_img, cam_params
                                )
                                
                                # Use best hypothesis for evaluation (but don't change training)
                                # This is for monitoring/validation purposes
                                best_idx = scores.argmax()
                                best_hypothesis = hypotheses_batch[best_idx]
                        except Exception as e:
                            # Validation failed, continue training
                            import warnings
                            warnings.warn(f"Hypothesis validation failed: {e}")
                
            # Backward and optimization
            self.optimizer_model.zero_grad(set_to_none=True)
            self.scaler_model.scale(loss).backward()
            self.scaler_model.step(self.optimizer_model)
            self.scaler_model.update()

            train_loss_epoch.append(loss.item())
        
        train_loss = np.mean(train_loss_epoch)
        return train_loss

    def validate_hypotheses(self, model, data_loader):
        model.eval()
        all_scores = []
        with torch.no_grad():
            for imgs, poses_gt in data_loader:
                imgs = imgs.to(self.args.device)
                mixture_distribution = model(imgs)
                
                # Sample hypotheses from the mixture distribution
                hypotheses = mixture_distribution.sample((10,)) # Sample 10 hypotheses
                
                # Validate hypotheses (this is a placeholder)
                for i in range(imgs.shape[0]):
                    scores = self.hypothesis_validator.validate(hypotheses[:, i], imgs[i])
                    all_scores.append(scores)
        
        return all_scores
