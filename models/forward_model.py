"""
Dual-Camera Snapshot Optical Forward Physics Model.

This module implements the complete mathematical forward simulation of the dual-camera
hyperspectral imaging platform described in Xie et al. (2022), "Dual camera snapshot
hyperspectral imaging system via physics-informed learning", Optics Express:

1. CASSI Imaging Branch (Grayscale Dispersive Snapshot):
   Spatial modulation via binary coded aperture mask followed by horizontal spectral
   dispersion through a prism and optical photon integration:
       f_{grayscale}(x', y) = \sum_{\lambda=1}^{N} h(x' - (\lambda - 1), y, \lambda) \cdot T(x' - (\lambda - 1), y)
   Sensor dimension:
       W_{disp} = W + (N - 1) * shift_step  (e.g., 256 + 30 = 286, 512 + 30 = 542).

2. Auxiliary Color Camera Branch:
   Spectral integration through the calibrated Camera Response Function (CRF) matrix K:
       f_{color}(c, x, y) = \sum_{\lambda=1}^{N} h(x, y, \lambda) \cdot K(c, \lambda)
   Sensor dimension: [B, 3, H, W].

3. Optical Inversion & Spatial Unshifting:
   Reverses dispersive displacement along horizontal x-axis:
       \hat{h}_{aligned}(x, y, \lambda) = f_{meas}(x + (\lambda - 1), y)
   Guarantees exact algebraic reversibility (\Delta < 1e-6) for unmodulated calibration signals.

Vectorization:
   All dispersion shifts and unshifting operations are implemented with vectorized
   PyTorch scatter/gather primitives, eliminating sequential Python loops and supporting
   full automatic differentiation with PyTorch AMP and CUDA.
"""

from typing import Optional, Union, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn

from .crf import CameraResponseFunction, project_rgb, get_crf_matrix
from .mask import CodedApertureMask, modulate_cube


class DualCameraForwardModel(nn.Module):
    """
    Differentiable Optical Forward Physics Pipeline for Dual-Camera Snapshot Hyperspectral Imaging.
    
    Attributes:
        num_bands: Number of spectral channels (N = 31 spanning 400-700 nm, 10 nm increments).
        shift_step: Horizontal spatial dispersion step per band (default 1 pixel/band).
        crf: Calibrated 3x31 Camera Response Function registered buffer.
    """
    def __init__(
        self,
        crf_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None,
        num_bands: int = 31,
        shift_step: int = 1,
        crf_profile: str = 'cave'
    ):
        super().__init__()
        self.num_bands = int(num_bands)
        self.shift_step = int(shift_step)

        if crf_matrix is not None:
            if isinstance(crf_matrix, np.ndarray):
                crf_t = torch.from_numpy(crf_matrix.astype(np.float32))
            elif isinstance(crf_matrix, torch.Tensor):
                crf_t = crf_matrix.clone().detach().to(dtype=torch.float32)
            else:
                raise TypeError(f"Unsupported crf_matrix type: {type(crf_matrix)}")
            if crf_t.shape != (3, self.num_bands):
                raise ValueError(
                    f"CRF matrix must have shape (3, {self.num_bands}), got {tuple(crf_t.shape)}"
                )
        else:
            crf_t = get_crf_matrix(crf_profile)

        self.register_buffer('crf', crf_t)

    def shift(
        self,
        cube: torch.Tensor,
        mask: Optional[Union[torch.Tensor, np.ndarray, CodedApertureMask]] = None
    ) -> torch.Tensor:
        """
        Applies coded aperture modulation and wavelength-dependent horizontal dispersion.
        
        Args:
            cube: Hyperspectral tensor of shape [B, 31, H, W] or [31, H, W].
            mask: Optional coded aperture mask [H, W], [1, 1, H, W], [B, 1, H, W] or CodedApertureMask.
            
        Returns:
            Shifted 3D cube [B, 31, H, W + 30] or [31, H, W + 30] without 2D channel summation.
        """
        is_batched = (cube.ndim == 4)
        if not is_batched:
            cube = cube.unsqueeze(0)
            
        B, C, H, W = cube.shape
        if C != self.num_bands:
            raise ValueError(f"Expected {self.num_bands} spectral bands, got {C}")

        # 1. Spatial modulation
        if mask is not None:
            if isinstance(mask, CodedApertureMask):
                mask = mask.tensor
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.to(device=cube.device, dtype=cube.dtype)
            
            if mask.ndim == 2:
                cube = cube * mask.view(1, 1, mask.shape[0], mask.shape[1])
            elif mask.ndim == 3:
                cube = cube * mask.unsqueeze(1)
            elif mask.ndim == 4:
                cube = cube * mask

        # 2. Vectorized horizontal spectral dispersion
        total_W = W + (C - 1) * self.shift_step
        k_idx = torch.arange(C, device=cube.device).unsqueeze(1) * self.shift_step  # [C, 1]
        x_idx = torch.arange(W, device=cube.device).unsqueeze(0)                     # [1, W]
        col_idx = (k_idx + x_idx).view(1, C, 1, W).expand(B, C, H, W)               # [B, C, H, W]

        # Scatter modulated bands into horizontally dispersed tensor
        shifted = torch.zeros(
            B, C, H, total_W,
            device=cube.device,
            dtype=cube.dtype
        ).scatter(dim=-1, index=col_idx, src=cube)

        return shifted if is_batched else shifted.squeeze(0)

    def forward_cassi(
        self,
        cube: torch.Tensor,
        mask: Optional[Union[torch.Tensor, np.ndarray, CodedApertureMask]] = None,
        return_shifted: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Simulates the CASSI grayscale branch forward imaging process.
        
        Modulates the scene with coded aperture mask T(x, y), horizontally shifts each band k
        by k * shift_step pixels, and sums across all N bands onto the 2D detector.

        Args:
            cube: Hyperspectral cube of shape [B, 31, H, W] or [31, H, W] in [0.0, 1.0].
            mask: Optional coded aperture mask [H, W], [B, 1, H, W] or CodedApertureMask.
            return_shifted: If True, returns tuple of (2D measurement, 3D dispersed tensor).
            
        Returns:
            meas: 2D compressed measurement [B, 1, H, W + 30] or [1, H, W + 30].
            shifted (optional): Dispersed 3D cube [B, 31, H, W + 30] or [31, H, W + 30].
        """
        is_batched = (cube.ndim == 4)
        shifted = self.shift(cube, mask=mask)
        
        if not is_batched:
            shifted_batched = shifted.unsqueeze(0)
        else:
            shifted_batched = shifted
            
        # Detector spectral integration (summation along channel dimension)
        meas = shifted_batched.sum(dim=1, keepdim=True)
        
        if not is_batched:
            meas = meas.squeeze(0)
            
        if return_shifted:
            return meas, shifted
        return meas

    def forward_rgb(
        self,
        cube: torch.Tensor,
        crf: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Simulates the auxiliary color camera branch forward imaging process.
        
        Performs spectral integration using the calibrated 3x31 CRF matrix:
            f_{color}(c, x, y) = \sum_{\lambda=1}^{N} h(x, y, \lambda) \cdot K(c, \lambda)

        Args:
            cube: Hyperspectral cube of shape [B, 31, H, W] or [31, H, W].
            crf: Optional CRF matrix [3, 31]. If None, uses registered self.crf.
            
        Returns:
            RGB image tensor [B, 3, H, W] or [3, H, W].
        """
        crf_mat = crf if crf is not None else self.crf
        if crf_mat is None:
            raise ValueError("No CRF matrix registered or provided.")
        return project_rgb(cube, crf_mat)

    @staticmethod
    def unshift_cassi(
        meas: Union[torch.Tensor, np.ndarray],
        num_bands: int = 31,
        original_w: Optional[int] = None,
        shift_step: int = 1
    ) -> torch.Tensor:
        """
        Reverses horizontal spectral dispersion, aligning dispersed channels back into spatial coincidence.
        
        Supports:
        - 2D/1-channel measurement: [B, 1, H, W_total] or [1, H, W_total]
          Extracts for band k the slice meas[:, :, :, k*shift_step : k*shift_step + original_w].
        - 3D dispersed tensor: [B, num_bands, H, W_total] or [num_bands, H, W_total]
          Extracts for band k the slice meas[:, k, :, k*shift_step : k*shift_step + original_w].
        - NumPy ndarrays and PyTorch tensors.

        Args:
            meas: Dispersed measurement or shifted tensor.
            num_bands: Number of spectral bands (default 31).
            original_w: Spatial width of undistorted scene (default W_total - (num_bands - 1) * shift_step).
            shift_step: Dispersion offset step per band (default 1).
            
        Returns:
            Aligned hyperspectral tensor [B, num_bands, H, original_w] or [num_bands, H, original_w].
        """
        is_numpy = isinstance(meas, np.ndarray)
        if is_numpy:
            meas_t = torch.from_numpy(meas)
        else:
            meas_t = meas

        is_batched = (meas_t.ndim == 4)
        if not is_batched:
            meas_t = meas_t.unsqueeze(0)

        B, C, H, total_W = meas_t.shape
        if original_w is None:
            W = total_W - (num_bands - 1) * shift_step
        else:
            W = int(original_w)
            
        if W <= 0 or total_W < W + (num_bands - 1) * shift_step:
            raise ValueError(
                f"Invalid dimensions: total_W={total_W}, original_w={W}, "
                f"num_bands={num_bands}, shift_step={shift_step}"
            )

        k_idx = torch.arange(num_bands, device=meas_t.device).unsqueeze(1) * shift_step  # [num_bands, 1]
        x_idx = torch.arange(W, device=meas_t.device).unsqueeze(0)                         # [1, W]
        col_idx = (k_idx + x_idx).view(1, num_bands, 1, W).expand(B, num_bands, H, W)     # [B, num_bands, H, W]

        if C == 1:
            # 1-channel compressive snapshot: expand and gather slice per band
            meas_expanded = meas_t.expand(B, num_bands, H, total_W)
            out = torch.gather(meas_expanded, dim=-1, index=col_idx)
        elif C == num_bands:
            # Multi-channel shifted cube: gather each band from its shifted offset
            out = torch.gather(meas_t, dim=-1, index=col_idx)
        else:
            raise ValueError(
                f"Expected channel dimension 1 or {num_bands}, got {C} (shape: {tuple(meas.shape)})"
            )

        return out if is_batched else out.squeeze(0)

    # Static alias for unshift_cassi
    unshift = unshift_cassi

    def forward(
        self,
        cube: torch.Tensor,
        mask: Optional[Union[torch.Tensor, np.ndarray, CodedApertureMask]] = None,
        crf: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full dual-camera forward observation pass.
        
        Args:
            cube: Hyperspectral scene radiance [B, 31, H, W] in [0.0, 1.0].
            mask: Optional coded aperture mask [H, W] or CodedApertureMask.
            crf: Optional CRF matrix [3, 31].
            
        Returns:
            Dictionary with:
                'cassi': 2D compressed measurement [B, 1, H, W + 30].
                'rgb': 3-channel color measurement [B, 3, H, W].
        """
        return {
            'cassi': self.forward_cassi(cube, mask=mask),
            'rgb': self.forward_rgb(cube, crf=crf)
        }

    def __repr__(self) -> str:
        return (
            f"DualCameraForwardModel(num_bands={self.num_bands}, "
            f"shift_step={self.shift_step}, crf_shape={tuple(self.crf.shape)})"
        )
