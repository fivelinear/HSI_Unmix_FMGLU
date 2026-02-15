import torch
import numpy as np
from pathlib import Path


class FMGLUUnmixer:
    """Hyperspectral unmixing inference interface"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def unmix(self, hsi_data, library_indices, batch_size=64):
        """
        Unmix hyperspectral image
        
        Args:
            hsi_data: (H, W, L) numpy array or (N, L) torch.Tensor
            library_indices: (k,) or (N, k) candidate endmember indices
            batch_size: inference batch size
            
        Returns:
            dict with endmembers, abundance, P, reconstruction, uncertainty
        """
        # Preprocess
        if isinstance(hsi_data, np.ndarray):
            hsi_data = torch.from_numpy(hsi_data).float()
            
        original_shape = hsi_data.shape
        
        # Flatten to (N, L)
        if len(original_shape) == 3:
            H, W, L = original_shape
            hsi_flat = hsi_data.view(-1, L)
        else:
            hsi_flat = hsi_data
            H, W = None, None
            
        N, L = hsi_flat.shape
        
        # Expand library_indices
        if isinstance(library_indices, np.ndarray):
            library_indices = torch.from_numpy(library_indices).long()
        if library_indices.dim() == 1:
            library_indices = library_indices.unsqueeze(0).repeat(N, 1)
            
        hsi_flat = hsi_flat.to(self.device)
        library_indices = library_indices.to(self.device)
        
        # Batch inference
        all_results = []
        
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch_hsi = hsi_flat[i:i+batch_size]
                batch_lib_idx = library_indices[i:i+batch_size]
                
                outputs = self.model(batch_hsi, batch_lib_idx)
                all_results.append(outputs)
        
        # Concatenate results
        results = {
            'endmembers': torch.cat([r['endmembers'] for r in all_results], dim=0),
            'abundance': torch.cat([r['abundance'] for r in all_results], dim=0),
            'P': torch.cat([r['P'] for r in all_results], dim=0),
            'reconstruction': torch.cat([r['reconstruction'] for r in all_results], dim=0),
        }
        
        # Restore spatial dimensions
        if H is not None and W is not None:
            results['abundance'] = results['abundance'].view(H, W, -1)
            results['P'] = results['P'].view(H, W, -1)
            results['reconstruction'] = results['reconstruction'].view(H, W, L)
            
        return results
    
    def unmix_with_uncertainty(self, hsi_data, library_indices, n_samples=10, batch_size=64):
        """
        Unmix with Monte Carlo uncertainty estimation
        
        Args:
            hsi_data: input hyperspectral data
            library_indices: candidate endmember indices
            n_samples: number of MC samples
            batch_size: inference batch size
            
        Returns:
            dict with mean and std for all outputs
        """
        self.model.train()  # Enable dropout for MC sampling
        
        samples = []
        for _ in range(n_samples):
            result = self.unmix(hsi_data, library_indices, batch_size)
            samples.append(result)
        
        self.model.eval()
        
        # Compute mean and std
        aggregated = {}
        for key in ['endmembers', 'abundance', 'P', 'reconstruction']:
            stacked = torch.stack([s[key] for s in samples])
            aggregated[key] = stacked.mean(dim=0)
            aggregated[f'{key}_std'] = stacked.std(dim=0)
            
        return aggregated
    
    def save_results(self, results, output_dir):
        """Save unmixing results to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                np.save(output_dir / f'{key}.npy', value.cpu().numpy())
                
        print(f"Results saved to {output_dir}")
