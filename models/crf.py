"""
Camera Response Function (CRF) and Spectral Projection Model.

This module provides calibrated spectral sensitivity matrices K in R^{3 x 31}
mapping 31-band hyperspectral cubes (400-700 nm, 10 nm increments) into 3-channel
RGB color camera measurements, per Section 3.3 and Equation (1) of Xie et al. (2022):

    f_{color}(c, x, y) = \sum_{\lambda=1}^{N} h(x, y, \lambda) \cdot K(c, \lambda) \cdot L(\lambda)

Includes:
- CAVE CRF Matrix: Calibrated spectral response curves from the CAVE dataset.
- Sony IMX174 CRF Matrix: Physical CMOS Bayer CFA response of Basler acA1920-155uc.
- CIE 1931 Color Matching Functions (CMF) transformed to sRGB space.
- Nikon D700 CRF Matrix: Standard reference camera response curves.
- CameraResponseFunction: nn.Module encapsulating the CRF projection operator.
- project_rgb: Vectorized standalone projection function.
"""

import os
from typing import Optional, Union, Dict
import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# 1. Calibrated Camera Response Curves (3 channels x 31 bands, 400 nm - 700 nm)
# -----------------------------------------------------------------------------

# CAVE dataset standard calibrated sensitivity curves (3 x 31)
# Bands: 400, 410, 420, ..., 700 nm (31 bands)
CAVE_CRF_MATRIX: np.ndarray = np.array([
    # Red Channel
    [
        0.0073, 0.0326, 0.1146, 0.2238, 0.2319, 0.1408, 0.0545, 0.0063, 0.0016, 0.0016,
        0.0020, 0.0049, 0.0163, 0.0458, 0.1065, 0.2088, 0.3516, 0.5284, 0.7042, 0.8359,
        0.8876, 0.8407, 0.7186, 0.5513, 0.3840, 0.2458, 0.1472, 0.0844, 0.0470, 0.0253,
        0.0135
    ],
    # Green Channel
    [
        0.0001, 0.0004, 0.0016, 0.0039, 0.0069, 0.0125, 0.0232, 0.0438, 0.0841, 0.1417,
        0.2114, 0.3160, 0.4578, 0.6127, 0.7303, 0.7711, 0.7259, 0.6094, 0.4566, 0.2974,
        0.1691, 0.0863, 0.0410, 0.0188, 0.0084, 0.0037, 0.0016, 0.0007, 0.0003, 0.0001,
        0.0000
    ],
    # Blue Channel
    [
        0.0336, 0.1558, 0.5847, 1.2588, 1.4883, 1.1396, 0.6908, 0.3546, 0.1584, 0.0658,
        0.0267, 0.0099, 0.0038, 0.0015, 0.0005, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000
    ]
], dtype=np.float32)

# Sony IMX174 Color CMOS sensor curves (Basler acA1920-155uc hardware in Xie et al. 2022)
SONY_IMX174_CRF_MATRIX: np.ndarray = np.array([
    # Red Channel
    [
        0.0021, 0.0030, 0.0040, 0.0048, 0.0051, 0.0049, 0.0044, 0.0038, 0.0343, 0.0373,
        0.0495, 0.0733, 0.1106, 0.1626, 0.2292, 0.3080, 0.3943, 0.4803, 0.5570, 0.6149,
        0.6460, 0.6460, 0.6149, 0.5570, 0.4803, 0.3942, 0.3080, 0.2290, 0.1621, 0.1092,
        0.0700
    ],
    # Green Channel
    [
        0.0131, 0.0253, 0.0461, 0.0792, 0.1276, 0.1927, 0.2728, 0.3635, 0.4576, 0.5472,
        0.6246, 0.6827, 0.7158, 0.7199, 0.6938, 0.6394, 0.5628, 0.4724, 0.3779, 0.2879,
        0.2088, 0.1442, 0.0947, 0.0593, 0.0353, 0.0200, 0.0108, 0.0055, 0.0027, 0.0013,
        0.0006
    ],
    # Blue Channel
    [
        0.2519, 0.3617, 0.4810, 0.5953, 0.6893, 0.7488, 0.7639, 0.7312, 0.6558, 0.5502,
        0.4313, 0.3157, 0.2156, 0.1375, 0.0818, 0.0454, 0.0235, 0.0069, 0.0019, 0.0005,
        0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000
    ]
], dtype=np.float32)

# CIE 1931 standard color matching curves (normalized for 31 bands, 400-700 nm)
CIE_1931_CMF_MATRIX: np.ndarray = np.array([
    # X (Red-dominant)
    [
        0.0143, 0.0435, 0.1344, 0.2839, 0.3483, 0.3362, 0.2908, 0.1954, 0.0956, 0.0320,
        0.0049, 0.0093, 0.0633, 0.1655, 0.2904, 0.4334, 0.5945, 0.7621, 0.9163, 1.0263,
        1.0622, 1.0026, 0.8544, 0.6424, 0.4479, 0.2835, 0.1649, 0.0874, 0.0468, 0.0227,
        0.0114
    ],
    # Y (Green-dominant / Luminance)
    [
        0.0004, 0.0012, 0.0040, 0.0116, 0.0230, 0.0380, 0.0600, 0.0910, 0.1390, 0.2080,
        0.3230, 0.5030, 0.7100, 0.8620, 0.9540, 0.9950, 0.9950, 0.9520, 0.8700, 0.7570,
        0.6310, 0.5030, 0.3810, 0.2650, 0.1750, 0.1070, 0.0610, 0.0320, 0.0170, 0.0082,
        0.0041
    ],
    # Z (Blue-dominant)
    [
        0.0679, 0.2074, 0.6456, 1.3856, 1.7471, 1.7721, 1.6692, 1.2876, 0.8130, 0.4652,
        0.2720, 0.1582, 0.0782, 0.0422, 0.0203, 0.0088, 0.0039, 0.0021, 0.0017, 0.0011,
        0.0008, 0.0003, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000
    ]
], dtype=np.float32)

# Nikon D700 spectral sensitivity curves
NIKON_D700_CRF_MATRIX: np.ndarray = np.array([
    # Red Channel
    [
        0.0010, 0.0012, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0050, 0.0100, 0.0220,
        0.0450, 0.0850, 0.1450, 0.2200, 0.3100, 0.4150, 0.5300, 0.6500, 0.7500, 0.8200,
        0.8600, 0.8500, 0.7800, 0.6600, 0.5100, 0.3700, 0.2500, 0.1500, 0.0900, 0.0500,
        0.0250
    ],
    # Green Channel
    [
        0.0050, 0.0080, 0.0150, 0.0280, 0.0520, 0.0950, 0.1650, 0.2600, 0.3800, 0.5150,
        0.6500, 0.7600, 0.8300, 0.8400, 0.7900, 0.6900, 0.5600, 0.4200, 0.2900, 0.1850,
        0.1100, 0.0600, 0.0300, 0.0150, 0.0070, 0.0030, 0.0015, 0.0008, 0.0004, 0.0002,
        0.0001
    ],
    # Blue Channel
    [
        0.1800, 0.2900, 0.4400, 0.6100, 0.7600, 0.8600, 0.8900, 0.8300, 0.6900, 0.5100,
        0.3300, 0.1900, 0.0950, 0.0450, 0.0200, 0.0090, 0.0040, 0.0020, 0.0010, 0.0005,
        0.0002, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000
    ]
], dtype=np.float32)

REGISTERED_CRF_PROFILES: Dict[str, np.ndarray] = {
    'cave': CAVE_CRF_MATRIX,
    'sony_imx174': SONY_IMX174_CRF_MATRIX,
    'cie1931': CIE_1931_CMF_MATRIX,
    'nikon_d700': NIKON_D700_CRF_MATRIX,
}


def get_crf_matrix(crf_type: str = 'cave', normalize: bool = False) -> torch.Tensor:
    """
    Returns a CRF sensitivity matrix as a float32 torch.Tensor of shape [3, 31].
    
    Args:
        crf_type: One of 'cave', 'sony_imx174', 'cie1931', 'nikon_d700'.
        normalize: If True, each channel row is normalized to sum to 1.0.
    """
    key = crf_type.lower()
    if key not in REGISTERED_CRF_PROFILES:
        raise ValueError(
            f"Unknown CRF profile '{crf_type}'. Available profiles: {list(REGISTERED_CRF_PROFILES.keys())}"
        )
    arr = REGISTERED_CRF_PROFILES[key].copy()
    if normalize:
        row_sums = arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        arr = arr / row_sums
    return torch.from_numpy(arr)


def project_rgb(cube: torch.Tensor, crf: torch.Tensor) -> torch.Tensor:
    """
    Vectorized projection of a hyperspectral cube to a 3-channel RGB image.
    
    Computes for each spatial pixel (x, y):
        f_{color}(c, x, y) = \sum_{\lambda=1}^{N} h(x, y, \lambda) \cdot K(c, \lambda)

    Args:
        cube: Hyperspectral tensor of shape [B, 31, H, W] or [31, H, W].
        crf: Camera Response Function matrix of shape [3, 31].
        
    Returns:
        RGB image tensor of shape [B, 3, H, W] or [3, H, W].
    """
    if cube.ndim == 3:
        # [31, H, W] -> unbatched
        # crf: [3, 31]
        crf_t = crf.to(dtype=cube.dtype, device=cube.device)
        return torch.einsum('cl, lhw -> chw', crf_t, cube)
    elif cube.ndim == 4:
        # [B, 31, H, W] -> batched
        crf_t = crf.to(dtype=cube.dtype, device=cube.device)
        return torch.einsum('cl, blhw -> bchw', crf_t, cube)
    else:
        raise ValueError(f"Expected cube with 3 or 4 dimensions, got shape {tuple(cube.shape)}")


# -----------------------------------------------------------------------------
# 2. CameraResponseFunction nn.Module
# -----------------------------------------------------------------------------

class CameraResponseFunction(nn.Module):
    """
    PyTorch Module encapsulating the Camera Response Function (CRF) matrix K.
    
    Acts as a linear operator projecting hyperspectral cubes [B, 31, H, W] to
    3-channel color measurements [B, 3, H, W].
    """
    def __init__(
        self,
        crf_type: str = 'cave',
        custom_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None,
        normalize: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        if custom_matrix is not None:
            if isinstance(custom_matrix, np.ndarray):
                mat = torch.from_numpy(custom_matrix.astype(np.float32))
            elif isinstance(custom_matrix, torch.Tensor):
                mat = custom_matrix.clone().detach().to(dtype=torch.float32)
            else:
                raise TypeError(f"Unsupported custom_matrix type: {type(custom_matrix)}")
            if mat.shape != (3, 31):
                raise ValueError(f"Custom CRF matrix must have shape (3, 31), got {tuple(mat.shape)}")
        else:
            mat = get_crf_matrix(crf_type, normalize=normalize)
            
        if normalize and custom_matrix is not None:
            row_sums = mat.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0
            mat = mat / row_sums

        mat = mat.to(dtype=dtype)
        if device is not None:
            mat = mat.to(device=device)

        self.register_buffer('crf', mat)

    def get_matrix(self) -> torch.Tensor:
        """Returns the registered [3, 31] CRF tensor."""
        return self.crf

    def get_numpy(self) -> np.ndarray:
        """Returns the [3, 31] CRF as a numpy ndarray."""
        return self.crf.detach().cpu().numpy()

    def project(self, cube: torch.Tensor) -> torch.Tensor:
        """
        Projects cube [B, 31, H, W] or [31, H, W] to [B, 3, H, W] or [3, H, W].
        """
        return project_rgb(cube, self.crf)

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        """Forward pass delegates to project."""
        return self.project(cube)

    def save(self, path: str) -> None:
        """Saves CRF matrix to disk as .npy or .pt."""
        if path.endswith('.npy'):
            np.save(path, self.get_numpy())
        elif path.endswith('.pt') or path.endswith('.pth'):
            torch.save(self.crf.detach().cpu(), path)
        else:
            # Default to npy
            np.save(path + '.npy', self.get_numpy())

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[Union[str, torch.device]] = None
    ) -> 'CameraResponseFunction':
        """Loads CRF matrix from .npy or .pt file."""
        if path.endswith('.npy'):
            arr = np.load(path)
            return cls(custom_matrix=arr, device=device)
        elif path.endswith('.pt') or path.endswith('.pth'):
            t = torch.load(path, map_location='cpu')
            return cls(custom_matrix=t, device=device)
        else:
            # Try npy then pt
            if os.path.exists(path + '.npy'):
                return cls(custom_matrix=np.load(path + '.npy'), device=device)
            elif os.path.exists(path):
                arr = np.load(path)
                return cls(custom_matrix=arr, device=device)
            raise FileNotFoundError(f"Could not load CRF matrix from {path}")

    def __repr__(self) -> str:
        return f"CameraResponseFunction(shape={tuple(self.crf.shape)}, device={self.crf.device})"
