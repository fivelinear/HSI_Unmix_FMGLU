import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import numpy as np


class Stage2Trainer:
    """Three-stage training for Stage 2"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        
    def setup_optimizer(self, lr_encoder=1e-4, lr_flow=5e-4, lr_decoder=5e-4):
        """Setup optimizer with parameter groups"""
        self.optimizer = AdamW([
            {'params': self.model.velocity_net.parameters(), 'lr': lr_flow},
            {'params': self.model.P_estimator.parameters(), 'lr': lr_decoder},
            {'params': self.model.abundance_estimator.parameters(), 'lr': lr_decoder},
            {'params': self.model.condition_proj.parameters(), 'lr': lr_flow},
            {'params': self.model.uncertainty_net.parameters(), 'lr': lr_decoder},
        ], weight_decay=1e-3)
        
    def flow_matching_loss(self, x0, x1, condition):
        """
        Flow Matching loss: L_FM = ||v_theta(x_t, t, c) - u_t||^2
        x0: (B, L) noise ~ N(0, I)
        x1: (B, L) target spectrum
        condition: (B, cond_dim)
        """
        B = x0.shape[0]
        t = torch.rand(B, device=x0.device)
        
        t_expanded = t.view(-1, 1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        ut = x1 - x0  # conditional vector field
        
        vt = self.model.velocity_net(xt, t, condition)
        loss = ((vt - ut) ** 2).mean()
        return loss
    
    def reconstruction_loss(self, X, X_hat):
        """MSE + SAD"""
        mse = ((X - X_hat) ** 2).mean()
        
        X_norm = X / (torch.norm(X, dim=-1, keepdim=True) + 1e-8)
        X_hat_norm = X_hat / (torch.norm(X_hat, dim=-1, keepdim=True) + 1e-8)
        cos_sim = (X_norm * X_hat_norm).sum(dim=-1)
        sad = torch.acos(cos_sim.clamp(-1, 1)).mean()
        
        return mse + 0.1 * sad
    
    def physical_constraint_loss(self, A):
        """ANC + ASC constraints"""
        anc = torch.relu(-A).pow(2).mean()
        asc = (A.sum(dim=-1) - 1).pow(2).mean()
        return anc + asc
    
    def variational_loss(self, outputs, target):
        """VI loss: -E_q[log p(X|E,A,P)] + KL(q||p)"""
        recon = outputs['reconstruction']
        unc = outputs['uncertainty']
        
        # Negative log likelihood
        nll = ((target - recon) ** 2).mean()
        
        # KL divergence for E, A, P
        kl_E = 0.5 * (unc['E_mean'].pow(2) + unc['E_logvar'].exp() - 1 - unc['E_logvar']).sum(dim=-1).mean()
        kl_A = 0.5 * (unc['A_mean'].pow(2) + unc['A_logvar'].exp() - 1 - unc['A_logvar']).sum(dim=-1).mean()
        kl_P = 0.5 * (unc['P_mean'].pow(2) + unc['P_logvar'].exp() - 1 - unc['P_logvar']).sum(dim=-1).mean()
        
        return nll + 0.1 * (kl_E + kl_A + kl_P)
    
    def train_stage2_flow(self, dataloader, epochs=1000):
        """
        Stage 2: Flow Matching pre-training
        Freeze encoder, train velocity network
        """
        print(f"=== Stage 2: Flow Matching Pre-training ({epochs} epochs) ===")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in dataloader:
                library_indices = batch['library_indices'].to(self.device)
                x1 = batch['target_spectra'].to(self.device)
                
                x0 = torch.randn_like(x1)
                conditions = self.model.get_library_condition(library_indices)
                conditions = conditions.squeeze(1)
                
                loss = self.flow_matching_loss(x0, x1, conditions)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
                
            if self.scheduler:
                self.scheduler.step()
    
    def train_stage3_joint(self, dataloader, epochs=800):
        """
        Stage 3: Joint fine-tuning
        Unfreeze encoder, end-to-end training
        """
        print(f"=== Stage 3: Joint Fine-tuning ({epochs} epochs) ===")
        
        # Unfreeze encoder
        self.model.unfreeze_encoder(self.optimizer, new_lr=1e-4)
        
        for epoch in range(epochs):
            self.model.current_epoch = epoch
            self.model.train()
            
            for batch in dataloader:
                hsi_pixels = batch['hsi'].to(self.device)
                library_indices = batch['library_indices'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Forward
                outputs = self.model(hsi_pixels, library_indices)
                
                # Compute losses
                loss_recon = self.reconstruction_loss(target, outputs['reconstruction'])
                loss_phys = self.physical_constraint_loss(outputs['abundance'])
                loss_vi = self.variational_loss(outputs, target)
                
                # Flow matching loss on generated endmembers
                z = torch.randn_like(outputs['endmembers'])
                lib_cond = self.model.get_library_condition(library_indices)
                loss_fm = self.flow_matching_loss(
                    z.view(-1, self.model.L),
                    outputs['endmembers'].view(-1, self.model.L),
                    lib_cond.view(-1, 128)
                )
                
                loss = loss_recon + 0.1 * loss_vi + 1.0 * loss_fm + 0.01 * loss_phys
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                
            if self.scheduler:
                self.scheduler.step()
    
    def train(self, dataloader, flow_epochs=1000, joint_epochs=800):
        """Full training pipeline"""
        self.setup_optimizer()
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=flow_epochs + joint_epochs
        )
        
        self.train_stage2_flow(dataloader, epochs=flow_epochs)
        self.train_stage3_joint(dataloader, epochs=joint_epochs)
        
        print("Training completed!")
        
    def save_checkpoint(self, path, epoch=None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")
        return checkpoint.get('epoch', 0)
