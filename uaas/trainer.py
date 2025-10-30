"""
Trainer for Uncertainty-Aware Adversarial Synthesis.
"""
from tqdm import tqdm
import numpy as np
import torch
from torch.cuda.amp import autocast

from rap import RVSWithDiscriminatorTrainer
from uaas.uaas_rap_net import UAASRAPNet
from uaas.sampler import UncertaintySampler
from uaas.loss import UncertaintyWeightedAdversarialLoss
from common.uncertainty import epistemic_uncertainty, aleatoric_uncertainty_regression

class UAASTrainer(RVSWithDiscriminatorTrainer):
    def __init__(self, args):
        super().__init__(args)
        # Override model with UAAS version
        self.model = UAASRAPNet(args).to(args.device)
        self.sampler = UncertaintySampler(self.renderer)
        self.adversarial_loss = UncertaintyWeightedAdversarialLoss()

    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def get_epistemic_uncertainty(self, model, imgs_batch, n_samples=10):
        model.train() # Enable dropout
        self.enable_dropout(model)
        
        poses = []
        with torch.no_grad():
            for _ in range(n_samples):
                _, pose_sample, _ = model(imgs_batch, return_feature=True)
                poses.append(pose_sample)
        
        poses = torch.stack(poses)
        return epistemic_uncertainty(poses)

    def train_epoch(self, epoch, poses_perturbed, imgs_perturbed):
        self.model.train()
        if self.args.freeze_batch_norm:
            self.model = self.freeze_bn_layer_train(self.model)

        train_loss_epoch = []

        selected_indexes = np.random.choice(self.dset_size, size=[self.dset_size], replace=False)

        i_batch = 0
        batch_size = self.args.batch_size
        device = self.args.device
        loss_weights = self.args.loss_weights
        
        for _ in tqdm(range(self.n_iters), desc=f'Epoch {epoch}'):
            if i_batch + batch_size > self.dset_size:
                break
            batch_indexes = selected_indexes[i_batch:i_batch + batch_size]
            i_batch += batch_size

            imgs_normed_batch = self.imgs_normed[batch_indexes].to(device, non_blocking=True)
            imgs_rendered_batch = self.imgs_rendered[batch_indexes].to(device, non_blocking=True)
            poses_batch = self.poses[batch_indexes].reshape(batch_size, 12).to(device, torch.float, non_blocking=True)

            with autocast(device, enabled=self.args.amp, dtype=self.args.amp_dtype):
                # Main model forward pass
                (features_target, features_rendered), (poses_predicted, log_var_predicted) = (
                    self.model(torch.cat((imgs_normed_batch, imgs_rendered_batch)), return_feature=True))
                
                # Uncertainty calculations
                epistemic_unc = self.get_epistemic_uncertainty(self.model, imgs_normed_batch)
                aleatoric_unc = aleatoric_uncertainty_regression(log_var_predicted)
                total_uncertainty = epistemic_unc + aleatoric_unc # Combine uncertainties
                
                # Use uncertainty to sample new data (placeholder)
                # new_poses, new_images = self.sampler.sample(...)

                # Pose loss with aleatoric uncertainty
                pose_loss = self.pose_loss(poses_predicted, poses_batch, aleatoric_unc)

                # --- Discriminator Training ---
                features_real = features_target.flatten(0, 1)
                features_fake = features_rendered.flatten(0, 1)
                
                disc_out_real = self.discriminator(features_real)
                disc_out_fake = self.discriminator(features_fake.detach())
                
                real_loss = self.adversarial_loss(disc_out_real, self.valid, torch.ones_like(self.valid))
                fake_loss = self.adversarial_loss(disc_out_fake, self.fake, torch.ones_like(self.fake))
                loss_disc = real_loss + fake_loss

            # Discriminator backward pass
            self.optimizer_disc.zero_grad(set_to_none=True)
            self.scaler_disc.scale(loss_disc).backward(retain_graph=True)
            self.scaler_disc.step(self.optimizer_disc)
            self.scaler_disc.update()

            with autocast(device, enabled=self.args.amp, dtype=self.args.amp_dtype):
                # --- Generator (RAPNet) Training ---
                loss_feature = self.feature_loss(features_rendered, features_target)

                # Adversarial loss weighted by uncertainty
                uncertainty_weights = torch.exp(-total_uncertainty.mean(dim=1)).detach()
                loss_generator = self.adversarial_loss(self.discriminator(features_fake), self.valid, uncertainty_weights.view(-1, 1))

                total_loss = (loss_weights[0] * pose_loss +
                              loss_weights[1] * loss_feature +
                              loss_weights[3] * loss_generator)
            
            # Generator backward pass
            self.optimizer_model.zero_grad(set_to_none=True)
            self.scaler_model.scale(total_loss).backward()
            self.scaler_model.step(self.optimizer_model)
            self.scaler_model.update()

            train_loss_epoch.append(total_loss.item())
        
        train_loss = np.mean(train_loss_epoch)
        return train_loss
