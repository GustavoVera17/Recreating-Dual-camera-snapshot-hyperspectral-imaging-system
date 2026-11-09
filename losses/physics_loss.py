"""
Closed-Loop Self-Supervised Physics-Informed Loss.

This module implements the self-supervised optimization objective described in Xie et al. (2022),
"Dual camera snapshot hyperspectral imaging system via physics-informed learning", Optics Express:

1. Pure Self-Supervision:
   The reconstruction network is optimized directly from physical observations without requiring
   paired ground-truth hyperspectral cubes during training.
   
2. CASSI Grayscale Loss (Eq. 5 in paper):
   Measures spectral dispersion fidelity between simulated CASSI measurement and physical sensor snapshot:
       Loss_grayscale = MSE(Forward_CASSI(pred_cube, mask), meas_cassi)

3. Color Camera Loss (Eq. 4 in paper):
   Measures spatial texture and colorimetric fidelity through the Camera Response Function (CRF):
       Loss_color = MSE(Forward_RGB(pred_cube, crf), meas_rgb)

4. Joint Physics Loss:
       Loss_total = Loss_grayscale + alpha * Loss_color (default alpha = 1.0)
"""

from typing import Optional, Union, Dict, Any, Tuple
import torch
import torch.nn as nn

from models.forward_model import DualCameraForwardModel


class LossResult(tuple):
    """
    Structured result container compatible with:
    - 3-element tuple unpacking: `loss_total, loss_cassi, loss_rgb = criterion(...)`
    - Named properties: `res.loss_total`, `res.loss_cassi`, `res.loss_rgb`, `res.loss_grayscale`, `res.loss_color`
    - Dictionary subscripting: `res['loss_total']`
    - Conversion to dict: `res.to_dict()`
    """
    def __new__(cls, loss_total: torch.Tensor, loss_cassi: torch.Tensor, loss_rgb: torch.Tensor):
        return super().__new__(cls, (loss_total, loss_cassi, loss_rgb))

    @property
    def loss_total(self) -> torch.Tensor:
        return self[0]

    @property
    def loss_cassi(self) -> torch.Tensor:
        return self[1]

    @property
    def loss_rgb(self) -> torch.Tensor:
        return self[2]

    @property
    def loss_grayscale(self) -> torch.Tensor:
        return self[1]

    @property
    def loss_color(self) -> torch.Tensor:
        return self[2]

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            'loss_total': self[0],
            'loss_cassi': self[1],
            'loss_rgb': self[2],
            'loss_grayscale': self[1],
            'loss_color': self[2],
        }

    def keys(self):
        return ['loss_total', 'loss_cassi', 'loss_rgb', 'loss_grayscale', 'loss_color']

    def __getitem__(self, item):
        if isinstance(item, str):
            mapping = {
                'loss_total': self[0],
                'loss_cassi': self[1],
                'loss_rgb': self[2],
                'loss_grayscale': self[1],
                'loss_color': self[2],
            }
            if item in mapping:
                return mapping[item]
            raise KeyError(f"Invalid key '{item}' in LossResult")
        return super().__getitem__(item)


class PhysicsLoss(nn.Module):
    """
    Closed-Loop Self-Supervised Physics Loss.
    
    Accepts:
        pred_cube: Reconstructed hyperspectral cube [B, 31, H, W] in [0.0, 1.0].
        meas_cassi: Real/observed CASSI measurement [B, 1, H, W + 30].
        meas_rgb: Real/observed RGB measurement [B, 3, H, W].
        mask: Binary coded aperture mask [H, W] or [B, 1, H, W].
        crf: Optional 3x31 Camera Response Function matrix.
        
    Returns:
        LossResult with (loss_total, loss_cassi, loss_rgb).
    """
    def __init__(
        self,
        forward_model: Optional[DualCameraForwardModel] = None,
        alpha: float = 1.0
    ):
        super().__init__()
        if forward_model is not None:
            self.forward_model = forward_model
        else:
            self.forward_model = DualCameraForwardModel()
        self.alpha = float(alpha)
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_cube: torch.Tensor,
        meas_cassi: torch.Tensor,
        meas_rgb: torch.Tensor,
        mask: Optional[Union[torch.Tensor, Any]] = None,
        crf: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> Union[LossResult, Dict[str, torch.Tensor]]:
        """
        Executes forward optical physics on predicted cube and computes self-supervised MSE losses.
        
        Zero reliance on ground-truth hyperspectral cubes.
        """
        # Forward simulation of CASSI branch: [B, 31, H, W] -> [B, 1, H, W + 30]
        sim_cassi = self.forward_model.forward_cassi(pred_cube, mask=mask)

        # Forward simulation of Color branch: [B, 31, H, W] -> [B, 3, H, W]
        sim_rgb = self.forward_model.forward_rgb(pred_cube, crf=crf)

        # Grayscale CASSI dispersion loss
        loss_cassi = self.mse(sim_cassi, meas_cassi)

        # Color camera CRF loss
        loss_rgb = self.mse(sim_rgb, meas_rgb)

        # Joint total loss
        loss_total = loss_cassi + self.alpha * loss_rgb

        result = LossResult(loss_total, loss_cassi, loss_rgb)
        if return_dict:
            return result.to_dict()
        return result


# Aliases
ClosedLoopPhysicsLoss = PhysicsLoss
DualPhysicsLoss = PhysicsLoss
