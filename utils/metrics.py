"""
Hyperspectral Image Reconstruction Evaluation Metrics Suite.

This module implements the quantitative evaluation suite described in Xie et al. (2022)
and Yamawaki et al. (2023):

1. Peak Signal-to-Noise Ratio (PSNR in dB):
   PSNR = 10 * log10(MAX_I^2 / MSE) = 20 * log10(MAX_I / sqrt(MSE))
   Capped at 100.0 dB for zero-error identity.

2. Structural Similarity Index (SSIM):
   Evaluated per spectral band across spatial dimensions (data_range=1.0) and averaged across channels.

3. Spectral Angle Mapper (SAM strictly in Radians):
   SAM(x, y) = arccos( (x . y) / (||x||_2 * ||y||_2) )
   Strictly evaluated in radians (orthogonal unit spectra yield pi/2 ≈ 1.5708 rad).
   Zero-norm safe with epsilon guard to prevent NaN gradients or division by zero.

4. Combined Evaluation:
   `evaluate_cube(pred, gt)` returns dictionary with 'psnr', 'ssim', and 'sam_rad'.
"""

from typing import Dict, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_metric


def calc_psnr(
    pred: Union[torch.Tensor, np.ndarray],
    gt: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    
    Args:
        pred: Predicted hyperspectral cube (Tensor or ndarray).
        gt: Ground-truth hyperspectral cube (Tensor or ndarray).
        max_val: Maximum dynamic range of signal (default 1.0).
        
    Returns:
        PSNR value in dB (clipped to 100.0 dB for identical tensors).
    """
    if isinstance(pred, torch.Tensor):
        p = pred.detach().float()
    else:
        p = torch.from_numpy(np.asarray(pred, dtype=np.float32))

    if isinstance(gt, torch.Tensor):
        g = gt.detach().float()
    else:
        g = torch.from_numpy(np.asarray(gt, dtype=np.float32))

    mse = torch.mean((p - g) ** 2).item()
    if mse <= 1e-12:
        return 100.0

    psnr = 10.0 * np.log10((float(max_val) ** 2) / mse)
    return float(min(psnr, 100.0))


def calc_ssim(
    pred: Union[torch.Tensor, np.ndarray],
    gt: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0
) -> float:
    """
    Computes mean Structural Similarity Index (SSIM) across all spectral channels.
    
    Args:
        pred: Predicted hyperspectral cube [B, C, H, W], [C, H, W], or ndarray.
        gt: Ground-truth hyperspectral cube [B, C, H, W], [C, H, W], or ndarray.
        max_val: Maximum dynamic range (default 1.0).
        
    Returns:
        Mean SSIM score in range [-1.0, 1.0].
    """
    if isinstance(pred, torch.Tensor):
        p = torch.clamp(pred.detach(), 0.0, float(max_val)).cpu().numpy()
    else:
        p = np.clip(np.asarray(pred, dtype=np.float32), 0.0, float(max_val))

    if isinstance(gt, torch.Tensor):
        g = torch.clamp(gt.detach(), 0.0, float(max_val)).cpu().numpy()
    else:
        g = np.clip(np.asarray(gt, dtype=np.float32), 0.0, float(max_val))

    # Handle batched 4D inputs [B, C, H, W]
    if p.ndim == 4:
        batch_ssims = []
        for b in range(p.shape[0]):
            p_hwc = p[b].transpose(1, 2, 0)
            g_hwc = g[b].transpose(1, 2, 0)
            val = ssim_metric(g_hwc, p_hwc, data_range=float(max_val), channel_axis=-1)
            batch_ssims.append(val)
        return float(np.mean(batch_ssims))

    # Handle 3D inputs [C, H, W]
    if p.ndim == 3:
        p_hwc = p.transpose(1, 2, 0)
        g_hwc = g.transpose(1, 2, 0)
        val = ssim_metric(g_hwc, p_hwc, data_range=float(max_val), channel_axis=-1)
        return float(val)

    # Handle 2D inputs [H, W]
    if p.ndim == 2:
        val = ssim_metric(g, p, data_range=float(max_val))
        return float(val)

    raise ValueError(f"Unsupported tensor dimensionality: {p.ndim}")


def calc_sam(
    pred: Union[torch.Tensor, np.ndarray],
    gt: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-7,
    in_degrees: bool = False
) -> float:
    """
    Computes the Spectral Angle Mapper (SAM) strictly in RADIANS.
    
    Formula:
        SAM(x, y) = arccos( (x . y) / (||x||_2 * ||y||_2) )
        
    Identity: SAM(x, x) = 0.0 rad.
    Orthogonal: SAM(x_orth, y_orth) = pi/2 ≈ 1.5708 rad.
    
    Args:
        pred: Predicted hyperspectral tensor [B, C, H, W], [C, H, W], or ndarray.
        gt: Ground-truth hyperspectral tensor of matching shape.
        eps: Numerical guard threshold for denominator (default 1e-7).
        in_degrees: If True, converts result to degrees (default False: strictly radians).
        
    Returns:
        Mean SAM angle across all spatial pixels in radians (or degrees).
    """
    if isinstance(pred, np.ndarray):
        p = torch.from_numpy(np.asarray(pred, dtype=np.float32))
    else:
        p = pred.detach().float()

    if isinstance(gt, np.ndarray):
        g = torch.from_numpy(np.asarray(gt, dtype=np.float32))
    else:
        g = gt.detach().float()

    if p.ndim == 4:
        # [B, C, H, W] -> [B, C, H*W]
        p_flat = p.view(p.shape[0], p.shape[1], -1)
        g_flat = g.view(g.shape[0], g.shape[1], -1)
    elif p.ndim == 3:
        # [C, H, W] -> [1, C, H*W]
        p_flat = p.unsqueeze(0).view(1, p.shape[0], -1)
        g_flat = g.unsqueeze(0).view(1, g.shape[0], -1)
    elif p.ndim == 2:
        # [C, N_pixels] -> [1, C, N_pixels]
        p_flat = p.unsqueeze(0)
        g_flat = g.unsqueeze(0)
    elif p.ndim == 1:
        # Single pixel spectrum [C] -> [1, C, 1]
        p_flat = p.view(1, -1, 1)
        g_flat = g.view(1, -1, 1)
    else:
        raise ValueError(f"Unsupported SAM tensor shape: {tuple(p.shape)}")

    # Promote to double precision to eliminate FP32 roundoff errors near cos(theta) = 1.0
    p_dbl = p_flat.double()
    g_dbl = g_flat.double()

    # Vectorized dot product and norms across spectral channel dimension C (dim=1)
    dot_prod = torch.sum(p_dbl * g_dbl, dim=1)  # [B, N_pixels]
    norm_p = torch.norm(p_dbl, p=2, dim=1)        # [B, N_pixels]
    norm_g = torch.norm(g_dbl, p=2, dim=1)        # [B, N_pixels]

    denom = norm_p * norm_g                        # [B, N_pixels]

    # Zero-norm safe: where either vector is all-zeros, angle is defined as 0.0 rad
    cos_theta = torch.ones_like(dot_prod)
    valid_mask = denom > float(eps)
    cos_theta[valid_mask] = dot_prod[valid_mask] / denom[valid_mask]

    # Numerical safeguard: clamp to valid domain [-1.0, 1.0]
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Compute angle in radians
    sam_rad = torch.acos(cos_theta)

    # If identical tensors within float precision, enforce exact zero
    if torch.allclose(p_dbl, g_dbl, atol=1e-7, rtol=1e-5):
        sam_rad.zero_()

    mean_sam = torch.mean(sam_rad).item()

    if in_degrees:
        return float(mean_sam * 180.0 / np.pi)
    return float(mean_sam)



def evaluate_cube(
    pred: Union[torch.Tensor, np.ndarray],
    gt: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0
) -> Dict[str, float]:
    """
    Evaluates PSNR, SSIM, and SAM on a predicted hyperspectral cube.
    
    Args:
        pred: Predicted cube [B, C, H, W] or [C, H, W].
        gt: Ground-truth cube of matching shape.
        max_val: Maximum reflectance range (default 1.0).
        
    Returns:
        Dictionary with keys 'psnr' (dB), 'ssim', 'sam_rad' (radians), and 'sam' (radians).
    """
    psnr = calc_psnr(pred, gt, max_val=max_val)
    ssim = calc_ssim(pred, gt, max_val=max_val)
    sam_rad = calc_sam(pred, gt)
    return {
        'psnr': psnr,
        'ssim': ssim,
        'sam_rad': sam_rad,
        'sam': sam_rad,
    }


# Standard function aliases for API consistency
calculate_psnr = calc_psnr
calculate_ssim = calc_ssim
calculate_sam = calc_sam
compute_psnr = calc_psnr
compute_ssim = calc_ssim
compute_sam = calc_sam
