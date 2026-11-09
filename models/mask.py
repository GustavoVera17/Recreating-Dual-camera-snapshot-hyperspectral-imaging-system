"""
Coded Aperture Mask Model and Spatial Modulation Operator.

This module models the lithographed binary Bernoulli coded aperture mask plate
used in the CASSI imaging branch of the dual-camera system, per Section 2 and
Section 3.1 of Xie et al. (2022).

Hardware characteristics:
- Lithographed Chrome on CaF2 substrate.
- Binary transmittance T(x, y) in {0.0, 1.0}.
- Theoretical Bernoulli transmittance probability p = 0.5.
- Standard scene aperture dimensions: 512 x 512 pixels.
- Spatial modulation equation:
      H_{mod}(x, y, \lambda) = H(x, y, \lambda) \odot T(x, y)
"""

import os
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def generate_bernoulli_mask(
    shape: Tuple[int, int] = (512, 512),
    p: float = 0.5,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generates a 2D binary Bernoulli random mask plate of shape (H, W).
    
    Args:
        shape: Spatial resolution (H, W), default (512, 512).
        p: Transmittance probability, default 0.5.
        seed: Random seed for deterministic reproducibility, default 42.
        
    Returns:
        np.ndarray: Float32 array of shape (H, W) with values in {0.0, 1.0}.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
        mask = rng.binomial(1, p, size=shape).astype(np.float32)
    else:
        mask = np.random.binomial(1, p, size=shape).astype(np.float32)
    return mask


def modulate_cube(
    cube: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Applies spatial modulation by the coded aperture mask:
        H_{mod} = H \odot T

    Supports PyTorch tensors and NumPy ndarrays with shapes:
    - [B, C, H, W]
    - [C, H, W]
    - [H, W, C]
    - [H, W]
    """
    if isinstance(cube, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=cube.dtype, device=cube.device)
        else:
            mask = mask.to(dtype=cube.dtype, device=cube.device)

        if cube.ndim == 4:
            # [B, C, H, W]
            if mask.ndim == 2:
                # [H, W] -> [1, 1, H, W]
                return cube * mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3:
                # [1, H, W] or [B, H, W]
                return cube * mask.unsqueeze(1)
            elif mask.ndim == 4:
                return cube * mask
            else:
                raise ValueError(f"Incompatible mask shape {tuple(mask.shape)} for 4D cube {tuple(cube.shape)}")
        elif cube.ndim == 3:
            # [C, H, W]
            if mask.ndim == 2:
                return cube * mask.unsqueeze(0)
            elif mask.ndim == 3:
                return cube * mask
            else:
                raise ValueError(f"Incompatible mask shape {tuple(mask.shape)} for 3D cube {tuple(cube.shape)}")
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {cube.ndim}D tensor")

    elif isinstance(cube, np.ndarray):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask, dtype=cube.dtype)

        if cube.ndim == 4:
            # [B, C, H, W]
            if mask.ndim == 2:
                return cube * mask[np.newaxis, np.newaxis, :, :]
            elif mask.ndim == 4:
                return cube * mask
        elif cube.ndim == 3:
            # Could be [C, H, W] or [H, W, C]
            if cube.shape[0] == mask.shape[0] and cube.shape[1] == mask.shape[1]:
                # [H, W, C]
                return cube * mask[:, :, np.newaxis]
            elif cube.shape[1] == mask.shape[0] and cube.shape[2] == mask.shape[1]:
                # [C, H, W]
                return cube * mask[np.newaxis, :, :]
            else:
                raise ValueError(f"Cannot align mask {mask.shape} with cube {cube.shape}")
        else:
            raise ValueError(f"Unsupported numpy array dimensions: {cube.ndim}")
    else:
        raise TypeError(f"Unsupported cube type: {type(cube)}")


class _CallableArray(np.ndarray):
    """Wrapper that allows array to be accessed as both property and method."""
    def __call__(self) -> np.ndarray:
        return self


class CodedApertureMask:
    """
    Manages the physical coded aperture mask plate for the CASSI system.
    
    Provides:
    - Bernoulli mask initialization (default shape 512x512, p=0.5, seed=42).
    - Loading and saving of calibration mask arrays (.npy or .pt).
    - Spatial patch extraction (random, centered, or arbitrary coordinates).
    - Element-wise spatial modulation: H_{mod} = H * T.
    """
    def __init__(
        self,
        mask_path: Optional[str] = None,
        mask_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
        shape: Tuple[int, int] = (512, 512),
        p: float = 0.5,
        seed: Optional[int] = 42,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32
    ):
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.dtype = dtype

        if mask_path is not None and os.path.exists(mask_path):
            self._mask_tensor = self._load_from_path(mask_path)
        elif mask_array is not None:
            if isinstance(mask_array, np.ndarray):
                self._mask_tensor = torch.from_numpy(mask_array.astype(np.float32)).to(self.dtype)
            elif isinstance(mask_array, torch.Tensor):
                self._mask_tensor = mask_array.clone().detach().to(dtype=self.dtype)
            else:
                raise TypeError(f"Unsupported mask_array type: {type(mask_array)}")
        else:
            # Generate deterministic Bernoulli binary mask
            raw_np = generate_bernoulli_mask(shape=shape, p=p, seed=seed)
            self._mask_tensor = torch.from_numpy(raw_np).to(self.dtype)

        self._mask_tensor = self._mask_tensor.to(device=self.device)
        self.H, self.W = self._mask_tensor.shape[-2], self._mask_tensor.shape[-1]

    def _load_from_path(self, path: str) -> torch.Tensor:
        if path.endswith('.npy'):
            arr = np.load(path).astype(np.float32)
            return torch.from_numpy(arr).to(self.dtype)
        elif path.endswith('.pt') or path.endswith('.pth'):
            t = torch.load(path, map_location='cpu')
            if isinstance(t, torch.Tensor):
                return t.to(dtype=self.dtype)
            raise ValueError(f"Loaded object from {path} is not a Tensor")
        else:
            # Try npy then pt
            if os.path.exists(path + '.npy'):
                return torch.from_numpy(np.load(path + '.npy').astype(np.float32)).to(self.dtype)
            arr = np.load(path).astype(np.float32)
            return torch.from_numpy(arr).to(self.dtype)

    @property
    def tensor(self) -> torch.Tensor:
        """Returns the mask as a 2D torch.Tensor of shape [H, W]."""
        return self._mask_tensor

    def get_tensor(self) -> torch.Tensor:
        """Returns the mask as a 2D torch.Tensor of shape [H, W]."""
        return self._mask_tensor

    @property
    def numpy(self) -> np.ndarray:
        """Returns the mask as a 2D float32 numpy array of shape (H, W). Supports both .numpy and .numpy()."""
        return self._mask_tensor.detach().cpu().numpy().view(_CallableArray)

    def get_numpy(self) -> np.ndarray:
        """Returns the mask as a 2D float32 numpy array of shape (H, W)."""
        return self._mask_tensor.detach().cpu().numpy()

    def to_numpy(self) -> np.ndarray:
        """Returns the mask as a 2D float32 numpy array of shape (H, W)."""
        return self._mask_tensor.detach().cpu().numpy()

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.H, self.W)

    def to(self, device: Union[str, torch.device]) -> 'CodedApertureMask':
        """Moves mask tensor to specified device."""
        self.device = torch.device(device)
        self._mask_tensor = self._mask_tensor.to(device=self.device)
        return self

    def get_patch(
        self,
        h_start: int,
        w_start: int,
        patch_h: int = 256,
        patch_w: int = 256
    ) -> torch.Tensor:
        """
        Slices a rectangular spatial patch from coordinates [h_start, w_start].
        
        Args:
            h_start: Vertical start index (0 <= h_start <= H - patch_h).
            w_start: Horizontal start index (0 <= w_start <= W - patch_w).
            patch_h: Height of crop.
            patch_w: Width of crop.
            
        Returns:
            torch.Tensor: Shape [patch_h, patch_w].
        """
        if h_start < 0 or h_start + patch_h > self.H:
            raise ValueError(f"h_start {h_start} with patch_h {patch_h} exceeds mask height {self.H}")
        if w_start < 0 or w_start + patch_w > self.W:
            raise ValueError(f"w_start {w_start} with patch_w {patch_w} exceeds mask width {self.W}")
        return self._mask_tensor[h_start : h_start + patch_h, w_start : w_start + patch_w]

    def get_center_patch(
        self,
        patch_h: int = 256,
        patch_w: int = 256
    ) -> torch.Tensor:
        """
        Slices the center crop of size [patch_h, patch_w] from the mask plate.
        """
        h_start = (self.H - patch_h) // 2
        w_start = (self.W - patch_w) // 2
        return self.get_patch(h_start, w_start, patch_h, patch_w)

    def get_random_patch(
        self,
        patch_h: int = 256,
        patch_w: int = 256,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Slices a random patch of size [patch_h, patch_w] and returns (patch, h_start, w_start).
        """
        max_h = self.H - patch_h
        max_w = self.W - patch_w
        if max_h < 0 or max_w < 0:
            raise ValueError(f"Patch dimensions ({patch_h}, {patch_w}) exceed mask size ({self.H}, {self.W})")

        if seed is not None:
            rng = np.random.RandomState(seed)
            h_start = int(rng.randint(0, max_h + 1))
            w_start = int(rng.randint(0, max_w + 1))
        else:
            h_start = int(np.random.randint(0, max_h + 1))
            w_start = int(np.random.randint(0, max_w + 1))

        patch = self.get_patch(h_start, w_start, patch_h, patch_w)
        return patch, h_start, w_start

    def modulate(
        self,
        cube: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multiplies cube by mask: H_{mod} = H * T.
        
        If mask is None, uses self._mask_tensor (or crops to cube spatial size if needed).
        """
        if mask is None:
            mask = self._mask_tensor
            # If cube spatial dimensions are smaller than mask plate, default to center crop
            cube_h, cube_w = cube.shape[-2], cube.shape[-1]
            if (cube_h, cube_w) != (self.H, self.W):
                mask = self.get_center_patch(cube_h, cube_w)

        return modulate_cube(cube, mask)

    def save(self, path: str) -> None:
        """Saves mask to disk as .npy or .pt."""
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if path.endswith('.npy'):
            np.save(path, self.numpy)
        elif path.endswith('.pt') or path.endswith('.pth'):
            torch.save(self._mask_tensor.detach().cpu(), path)
        else:
            np.save(path + '.npy', self.numpy)

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[Union[str, torch.device]] = None
    ) -> 'CodedApertureMask':
        """Factory method loading a mask plate from file."""
        return cls(mask_path=path, device=device)

    def __repr__(self) -> str:
        mean_p = float(self._mask_tensor.mean().item())
        return (
            f"CodedApertureMask(shape={tuple(self.shape)}, "
            f"transmittance_p={mean_p:.4f}, device={self.device})"
        )
