import torch
import torch.nn as nn
from pathlib import Path
import json


class SpectralFeatureEncoder(nn.Module):
    """1D-CNN + Transformer hybrid encoder from Stage 1"""
    
    def __init__(self, L, hidden_dim=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.L = L
        self.hidden_dim = hidden_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, hidden_dim//2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, mask=None):
        # x: (B, L)
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.conv_layers(x)  # (B, hidden_dim, L)
        
        x = x.permute(2, 0, 1)  # (L, B, hidden_dim)
        x_att, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + x_att)
        x_ffn = self.ffn(x)
        x = self.norm2(x + x_ffn)
        
        x = x.mean(dim=0)  # (B, hidden_dim)
        return x, None


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embedding for Flow Matching"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        # t: (B,)
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class VelocityNetwork(nn.Module):
    """Flow Matching velocity field network"""
    
    def __init__(self, L, cond_dim=128, time_embed_dim=128, hidden_dim=256):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_embed_dim)
        
        input_dim = L + time_embed_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, L)
        )
        
    def forward(self, x, t, condition):
        # x: (B, L), t: (B,), condition: (B, cond_dim)
        t_emb = self.time_embed(t)
        v = self.net(torch.cat([x, t_emb, condition], dim=-1))
        return v


class UncertaintyEstimator(nn.Module):
    """Variational inference for uncertainty quantification"""
    
    def __init__(self, feature_dim, n_endmembers, L):
        super().__init__()
        
        # Endmember uncertainty: output mean and log_var
        self.E_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_endmembers * L * 2)  # mean + log_var
        )
        
        # Abundance uncertainty
        self.A_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_endmembers * 2)
        )
        
        # P uncertainty
        self.P_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, L * 2)
        )
        
    def forward(self, features):
        # features: (B, feature_dim)
        B = features.shape[0]
        
        E_params = self.E_head(features)
        A_params = self.A_head(features)
        P_params = self.P_head(features)
        
        return {
            'E_mean': E_params[..., 0::2],
            'E_logvar': E_params[..., 1::2],
            'A_mean': A_params[..., 0::2],
            'A_logvar': A_params[..., 1::2],
            'P_mean': P_params[..., 0::2],
            'P_logvar': P_params[..., 1::2],
        }


class Stage2Model(nn.Module):
    """
    Stage 2: Flow Matching Guided Library-based Unmixing
    Loads Stage 1 assets and implements FM-based unmixing
    """
    
    def __init__(self, stage1_output_dir, n_endmembers, freeze_epochs=200):
        super().__init__()
        
        stage1_dir = Path(stage1_output_dir)
        
        # Load Stage 1 encoder
        encoder_ckpt = torch.load(stage1_dir / 'encoder_final.pth', weights_only=False)
        self.encoder = SpectralFeatureEncoder(**encoder_ckpt['config'])
        self.encoder.load_state_dict(encoder_ckpt['encoder_state_dict'])
        
        self.L = encoder_ckpt['config']['L']
        self.hidden_dim = encoder_ckpt['config']['hidden_dim']
        self.n_endmembers = n_endmembers
        
        # Freeze strategy: first 200 epochs
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Load precomputed library features as buffer
        lib_data = torch.load(stage1_dir / 'library_features.pt', weights_only=False)
        self.register_buffer('library_features', lib_data['features'])  # (14530, 256)
        self.library_names = lib_data['spectra_names']
        self.n_library = lib_data['n_spectra']
        
        # Load wavelength config
        with open(stage1_dir / 'wavelength_config.json', 'r') as f:
            self.wl_config = json.load(f)
        
        # Stage 2 components
        self.condition_proj = nn.Linear(256, 128)
        
        self.velocity_net = VelocityNetwork(
            L=self.L,
            cond_dim=128,
            time_embed_dim=128,
            hidden_dim=256
        )
        
        self.P_estimator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.L),
            nn.Sigmoid()
        )
        
        self.abundance_estimator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_endmembers),
            nn.Softmax(dim=-1)
        )
        
        self.uncertainty_net = UncertaintyEstimator(256, n_endmembers, self.L)
        
    def get_library_condition(self, library_indices):
        # library_indices: (B, k) -> returns (B, k, 128)
        B, k = library_indices.shape
        flat_indices = library_indices.view(-1)
        raw_features = self.library_features[flat_indices]  # (B*k, 256)
        conditions = self.condition_proj(raw_features)  # (B*k, 128)
        return conditions.view(B, k, 128)
    
    def unfreeze_encoder(self, optimizer, new_lr=1e-4):
        for param in self.encoder.parameters():
            param.requires_grad = True
        optimizer.add_param_group({'params': self.encoder.parameters(), 'lr': new_lr})
        
    def flow_matching_sample(self, z, conditions, num_steps=4):
        # z: (B, k, L), conditions: (B, k, 128) -> E: (B, k, L)
        B, k, L = z.shape
        dt = 1.0 / num_steps
        
        x = z.view(B * k, L)
        cond_flat = conditions.view(B * k, -1)
        
        for i in range(num_steps):
            t = torch.ones(B * k, device=x.device) * (i * dt)
            v = self.velocity_net(x, t, cond_flat)
            x = x + dt * v
            
        return x.view(B, k, L)
    
    def mlm_forward(self, E, A, P):
        # E: (B, k, L), A: (B, N, k), P: (B, L) -> X_hat: (B, N, L)
        # Linear mixing: Y = E @ A^T
        Y = torch.bmm(E.transpose(1, 2), A.transpose(1, 2))  # (B, L, N)
        
        # MLM nonlinear: X = (1-P) * Y / (1 - P * Y)
        P_expanded = P.unsqueeze(-1)  # (B, L, 1)
        numerator = (1 - P_expanded) * Y
        denominator = 1 - P_expanded * Y
        X_hat = numerator / (denominator + 1e-8)
        
        return X_hat.transpose(1, 2)  # (B, N, L)
    
    def forward(self, hsi_pixels, library_indices):
        # hsi_pixels: (B, L), library_indices: (B, k)
        B, L = hsi_pixels.shape
        
        # Extract features with optional gradient
        with torch.set_grad_enabled(
            not self.training or self.current_epoch >= self.freeze_epochs
        ):
            img_features, _ = self.encoder(hsi_pixels, mask=None)  # (B, 256)
        
        # Get library conditions
        lib_conditions = self.get_library_condition(library_indices)  # (B, k, 128)
        
        # Flow Matching generate endmembers
        z = torch.randn(B, self.n_endmembers, self.L, device=hsi_pixels.device)
        E = self.flow_matching_sample(z, lib_conditions, num_steps=4)  # (B, k, L)
        
        # Estimate abundance
        A = self.abundance_estimator(img_features)  # (B, k)
        
        # Estimate P
        P = self.P_estimator(img_features)  # (B, L)
        
        # MLM reconstruction
        A_expanded = A.unsqueeze(1)  # (B, 1, k)
        X_hat = self.mlm_forward(E, A_expanded, P)  # (B, 1, L)
        
        # Uncertainty
        uncertainty = self.uncertainty_net(img_features)
        
        return {
            'endmembers': E,
            'abundance': A,
            'P': P,
            'reconstruction': X_hat.squeeze(1),
            'uncertainty': uncertainty
        }
