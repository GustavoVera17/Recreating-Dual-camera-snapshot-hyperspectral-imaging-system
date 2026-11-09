"""
Optical and Reconstruction Models Package for Xie_Rep_2.

Exposes:
- DualCameraForwardModel: Optical forward simulation (CASSI dispersion + Color CRF).
- CameraResponseFunction: Spectral camera response curves and projection operator.
- CodedApertureMask: Physical binary Bernoulli coded aperture mask plate and modulation.
"""

from .crf import (
    CameraResponseFunction,
    project_rgb,
    get_crf_matrix,
    CAVE_CRF_MATRIX,
    SONY_IMX174_CRF_MATRIX,
    CIE_1931_CMF_MATRIX,
    NIKON_D700_CRF_MATRIX,
)

from .mask import (
    CodedApertureMask,
    generate_bernoulli_mask,
    modulate_cube,
)

from .forward_model import (
    DualCameraForwardModel,
)

# Aliases for flexible naming conventions across paper and code
ForwardModel = DualCameraForwardModel
CRF = CameraResponseFunction
Mask = CodedApertureMask

# Optional imports for modules implemented in subsequent milestones
try:
    from .sir_cnn import (
        SpectralImageReconstructionCNN,
        SIR_CNN,
        SIRCNN,
        Stage1DimensionalAscending,
    )
except ImportError:
    pass

try:
    from .hrnet_backbone import HRNetBackbone, BasicBlock, HRModule, HRStage
except ImportError:
    pass

__all__ = [
    'DualCameraForwardModel',
    'ForwardModel',
    'CameraResponseFunction',
    'CRF',
    'CodedApertureMask',
    'Mask',
    'SpectralImageReconstructionCNN',
    'SIR_CNN',
    'SIRCNN',
    'HRNetBackbone',
    'Stage1DimensionalAscending',
    'generate_bernoulli_mask',
    'modulate_cube',
    'project_rgb',
    'get_crf_matrix',
    'CAVE_CRF_MATRIX',
    'SONY_IMX174_CRF_MATRIX',
    'CIE_1931_CMF_MATRIX',
    'NIKON_D700_CRF_MATRIX',
]

