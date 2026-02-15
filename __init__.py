"""
FMGLU-Net Stage 2: Flow Matching Guided Library-based Unmixing

Usage:
    from stage2.model import Stage2Model
    from stage2.train import Stage2Trainer
    from stage2.unmix import FMGLUUnmixer
    
    # Load model
    model = Stage2Model(
        stage1_output_dir='./stage1_outputs',
        n_endmembers=5,
        freeze_epochs=200
    )
    
    # Train
    trainer = Stage2Trainer(model, device='cuda')
    trainer.train(dataloader, flow_epochs=1000, joint_epochs=800)
    
    # Unmix
    unmixer = FMGLUUnmixer(model)
    results = unmixer.unmix(hsi_data, library_indices)
"""

from .model import Stage2Model, SpectralFeatureEncoder, VelocityNetwork
from .train import Stage2Trainer
from .unmix import FMGLUUnmixer

__all__ = [
    'Stage2Model',
    'SpectralFeatureEncoder',
    'VelocityNetwork',
    'Stage2Trainer',
    'FMGLUUnmixer',
]
