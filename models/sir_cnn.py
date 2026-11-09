"""
Spectral Image Reconstruction CNN (SIR-CNN).

This module implements the complete reconstruction network architecture described in Xie et al. (2022),
"Dual camera snapshot hyperspectral imaging system via physics-informed learning", Optics Express:

1. Stage 1 (Dimensional Ascension Layer):
   Spatial-spectral de-multiplexing and dimension reduction convolution:
       nn.Conv2d(1, 31, kernel_size=(1, 31), stride=1, padding=0, bias=True) + ReLU
   Converts the 2D horizontally dispersed CASSI measurement [B, 1, H, W + 30]
   into a 3D spatio-spectral feature representation [B, 31, H, W].

2. Stage 2 (Parallel-Multiscale Refinement Backbone):
   HRNet parallel multi-resolution network maintaining 4 parallel streams (1x, 1/2x, 1/4x, 1/8x)
   with cross-scale fusion cascades and optional gradient checkpointing.

3. Multiscale Feature Aggregation Head:
   Bilinear upsampling of all 4 parallel streams to full spatial resolution (H x W),
   followed by channel concatenation (15 * C1 channels = 480 channels for W32).
   Refinement projection: Conv 1x1 -> BatchNorm -> ReLU -> Conv 3x3 -> 31 channels.

4. Physical Spectral Bounding:
   Strict output clamping `torch.clamp(out, min=0.0, max=1.0)` enforcing physical reflectance
   conservation and preventing negative spectral cancellation artifacts.
"""

from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrnet_backbone import HRNetBackbone, HRNET_CONFIGS, BasicBlock, HRModule, HRStage


class Stage1DimensionalAscending(nn.Module):
    """
    Stage 1: Dimensional Ascension & Dispersion Contraction Layer.
    
    Applies lambda convolution filters of size (1, lambda) with stride 1 and padding 0,
    followed by ReLU activation. Maps [B, 1, H, W + lambda - 1] to [B, lambda, H, W].
    """
    def __init__(self, in_channels: int = 1, num_bands: int = 31):
        super().__init__()
        self.conv1x_lambda = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_bands,
            kernel_size=(1, num_bands),
            stride=1,
            padding=0,
            bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv1x_lambda(x))

    def __getitem__(self, idx: int) -> nn.Module:
        if idx == 0:
            return self.conv1x_lambda
        elif idx == 1:
            return self.relu
        raise IndexError("Stage1DimensionalAscending has 2 layers: 0 (Conv2d) and 1 (ReLU)")

    def __len__(self) -> int:
        return 2

    def __iter__(self):
        return iter([self.conv1x_lambda, self.relu])



class SpectralImageReconstructionCNN(nn.Module):
    """
    Full Dual-Stage Hyperspectral Reconstruction Network (SIR-CNN).
    
    Stage 1: (1, lambda) unmixing convolution (286 -> 256 along width).
    Stage 2: 4-branch parallel multi-resolution HRNet backbone.
    Head: 15*C1 concatenation -> 1x1 conv -> BN -> ReLU -> 3x3 conv -> clamp([0.0, 1.0]).
    
    Attributes:
        stage1: Stage1DimensionalAscending layer.
        backbone: HRNetBackbone instance.
        head: Multi-scale aggregation projection head.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_bands: int = 31,
        hrnet_type: str = 'w32',
        c1: Optional[int] = None,
        use_checkpoint: bool = True
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_bands = int(num_bands)
        self.hrnet_type = str(hrnet_type)

        # Stage 1: Dimensional Ascension Layer
        self.stage1 = Stage1DimensionalAscending(
            in_channels=self.in_channels,
            num_bands=self.num_bands
        )
        # Convenient aliases for audit and legacy compatibility
        self.stage1_ascension = self.stage1
        self.dimensional_ascension = self.stage1

        # Stage 2: Parallel-Multiscale HRNet Refinement Backbone
        self.backbone = HRNetBackbone(
            in_channels=self.num_bands,
            c1=c1 if c1 is not None else HRNET_CONFIGS.get(self.hrnet_type.lower(), 32),
            hrnet_type=self.hrnet_type,
            use_checkpoint=use_checkpoint
        )
        self.stage2 = self.backbone
        self.channels = self.backbone.channels

        # Multiscale Aggregation & Refinement Head
        total_channels = self.backbone.total_channels  # 15 * c1 (480 for W32)
        self.head = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels, self.num_bands, kernel_size=3, padding=1, bias=True)
        )

    @property
    def use_checkpoint(self) -> bool:
        return self.backbone.use_checkpoint

    @use_checkpoint.setter
    def use_checkpoint(self, val: bool) -> None:
        self.backbone.set_use_checkpoint(val)

    def forward(self, meas: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs 3D hyperspectral cube from 2D dispersed CASSI snapshot.
        
        Args:
            meas: Dispersed measurement tensor [B, 1, H, W + num_bands - 1]
                  or [1, H, W + num_bands - 1] (e.g., [B, 1, 256, 286] or [B, 1, 512, 542]).
                  
        Returns:
            Reconstructed cube [B, num_bands, H, W] or [num_bands, H, W]
            strictly clamped within physical reflectance interval [0.0, 1.0].
        """
        is_batched = (meas.ndim == 4)
        if not is_batched:
            meas = meas.unsqueeze(0)

        B, C, H, W_total = meas.shape
        expected_min_w = self.num_bands
        if W_total < expected_min_w:
            raise ValueError(
                f"Input width ({W_total}) must be >= num_bands ({expected_min_w})"
            )

        # Stage 1: Dimensional Ascension (width: W_total -> W_total - num_bands + 1)
        x_stage1 = self.stage1(meas)

        # Stage 2: Parallel-Multiscale HRNet Backbone
        streams = self.backbone(x_stage1)

        # Multiscale aggregation: upsample all streams to H x W and concatenate
        fused = self.backbone.aggregate(streams)

        # Reconstruction head projection
        raw_pred = self.head(fused)

        # Hard physical spectral bounding
        out = torch.clamp(raw_pred, min=0.0, max=1.0)

        return out if is_batched else out.squeeze(0)


# Aliases for naming flexibility across codebase and paper citations
SIRCNN = SpectralImageReconstructionCNN
SIR_CNN = SpectralImageReconstructionCNN
