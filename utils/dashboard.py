"""
Headless Visual Dashboard Generator for Dual-Camera Snapshot Hyperspectral Imaging.

This module implements the 2x4 visual diagnostic dashboard generator standardized in
Yamawaki_Rep_2 and specified for Xie_Rep_2 in REVERSE_ENGINEERING_PLAN.md:

Subplot Layout (2 Rows x 4 Columns, figsize=(22, 11)):
-----------------------------------------------------------------------------------------
ROW 1: Spatial & Spectral Visual Reconstructions
[0, 0] CASSI 2D Measurement      : Grayscale compressed sensor snapshot [H, W + 30].
[0, 1] Ground Truth Band 15     : Central wavelength (~550 nm) grayscale reflectance [0, 1].
[0, 2] Predicted Band 15        : Model reconstruction central wavelength reflectance [0, 1].
[0, 3] Absolute Spatial Error   : Heatmap |pred - gt| with inferno colormap and colorbar.
-----------------------------------------------------------------------------------------
ROW 2: Quantitative Convergence Curves
[1, 0] Training Loss Curve       : Red line ('r-'), linear scale, grid enabled.
[1, 1] Validation PSNR Curve     : Blue line ('b-'), decibel scale (dB), grid enabled.
[1, 2] Validation SSIM Curve     : Green line ('g-'), [0, 1] scale, grid enabled.
[1, 3] Validation SAM (rad) Curve: Magenta line ('m-'), radians scale, grid enabled.
-----------------------------------------------------------------------------------------

Execution Environment:
- Uses headless Matplotlib ('Agg' backend) to prevent display server errors on headless servers.
- Saves output cleanly to 'results/dashboards/dashboard_ep{epoch:04d}.png'.
"""

import os
from typing import Optional, Union, Dict, List, Any
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _to_numpy_2d(tensor_or_array: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Converts a tensor or array to a 2D float32 numpy array."""
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor_or_array, dtype=np.float32)

    # Squeeze leading singleton dimensions
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr.squeeze(0)

    if arr.ndim > 2:
        # If still multi-channel (e.g. [C, H, W]), take channel 0
        arr = arr[0]

    return arr.astype(np.float32)


def _extract_band_2d(
    cube: Union[torch.Tensor, np.ndarray],
    band_idx: int = 15
) -> np.ndarray:
    """
    Extracts a single 2D spectral band from a 3D or 4D hyperspectral cube.
    
    Args:
        cube: Hyperspectral cube [B, C, H, W], [C, H, W], or [H, W, C].
        band_idx: Spectral band index (default 15 for ~550 nm).
        
    Returns:
        2D numpy array [H, W].
    """
    if isinstance(cube, torch.Tensor):
        arr = cube.detach().cpu().float().numpy()
    else:
        arr = np.asarray(cube, dtype=np.float32)

    # Batched [B, C, H, W]
    if arr.ndim == 4:
        arr = arr[0]

    # [C, H, W]
    if arr.ndim == 3:
        if arr.shape[0] <= 32:  # Channels first
            b_idx = min(band_idx, arr.shape[0] - 1)
            return arr[b_idx].astype(np.float32)
        elif arr.shape[2] <= 32:  # Channels last [H, W, C]
            b_idx = min(band_idx, arr.shape[2] - 1)
            return arr[:, :, b_idx].astype(np.float32)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    raise ValueError(f"Cannot extract band from shape: {arr.shape}")


def hsi_to_pseudo_rgb(cube: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Creates a normalized pseudo-RGB composite from a 31-band HSI cube.
    
    Uses bands 26 (Red ~660 nm), 14 (Green ~540 nm), and 5 (Blue ~450 nm).
    
    Returns:
        RGB image array [H, W, 3] in [0.0, 1.0].
    """
    if isinstance(cube, torch.Tensor):
        arr = cube.detach().cpu().float().numpy()
    else:
        arr = np.asarray(cube, dtype=np.float32)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[2] == 31:
        arr = np.transpose(arr, (2, 0, 1))

    r = arr[min(26, arr.shape[0] - 1)]
    g = arr[min(14, arr.shape[0] - 1)]
    b = arr[min(5, arr.shape[0] - 1)]

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    return rgb


class VisualDashboardGenerator:
    """
    Headless 2x4 visual dashboard generator.
    
    Attributes:
        output_dir: Destination directory for rendered dashboard images.
        figsize: Figure dimensions tuple in inches (default (22, 11)).
        dpi: Output image resolution in DPI (default 130).
    """
    def __init__(
        self,
        output_dir: Optional[str] = None,
        figsize: tuple = (22, 11),
        dpi: int = 130,
    ):
        if output_dir is not None:
            self.output_dir = os.path.abspath(output_dir)
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.output_dir = os.path.join(base_dir, "results", "dashboards")

        os.makedirs(self.output_dir, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi

    def generate(
        self,
        epoch: int,
        cassi_meas: Union[torch.Tensor, np.ndarray],
        gt_cube: Union[torch.Tensor, np.ndarray],
        pred_cube: Union[torch.Tensor, np.ndarray],
        history: Optional[Dict[str, List[Any]]] = None,
        scene_name: Optional[str] = None,
        band_idx: int = 15,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Renders and saves the 2x4 visual diagnostic dashboard.
        
        Args:
            epoch: Current training epoch index.
            cassi_meas: CASSI 2D measurement [B, 1, H, W+30] or [H, W+30].
            gt_cube: Ground-truth HSI cube [B, 31, H, W] or [31, H, W].
            pred_cube: Model reconstructed HSI cube [B, 31, H, W] or [31, H, W].
            history: Dictionary of metric histories with keys:
                     'loss' (or 'train_loss'), 'psnr' (or 'val_psnr'),
                     'ssim' (or 'val_ssim'), 'sam' (or 'val_sam').
            scene_name: Optional name of the validation scene.
            band_idx: Central spectral band index to visualize (default 15).
            save_path: Optional explicit output filepath. Defaults to
                       'results/dashboards/dashboard_ep{epoch:04d}.png'.
                       
        Returns:
            Absolute filesystem path of saved PNG file.
        """
        if save_path is None:
            filename = f"dashboard_ep{epoch:04d}.png"
            out_file = os.path.join(self.output_dir, filename)
        else:
            out_file = os.path.abspath(save_path)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

        if history is None:
            history = {}

        # 1. Prepare 2D spatial images
        meas_2d = _to_numpy_2d(cassi_meas)
        gt_band = np.clip(_extract_band_2d(gt_cube, band_idx), 0.0, 1.0)
        pred_band = np.clip(_extract_band_2d(pred_cube, band_idx), 0.0, 1.0)
        err_map = np.abs(gt_band - pred_band)

        # 2. Extract convergence histories
        def _get_series(keys: List[str]) -> List[float]:
            for k in keys:
                if k in history and len(history[k]) > 0:
                    return [float(x) for x in history[k]]
            return []

        loss_series = _get_series(['train_loss', 'loss'])
        psnr_series = _get_series(['val_psnr', 'psnr'])
        ssim_series = _get_series(['val_ssim', 'ssim'])
        sam_series = _get_series(['val_sam', 'sam'])

        # Create 2x4 Subplot Grid
        fig, axs = plt.subplots(2, 4, figsize=self.figsize)

        # Build figure suptitle
        title_parts = [f"Xie et al. (2022) Replication | Epoch {epoch:04d}"]
        if scene_name:
            title_parts.append(f"Scene: {scene_name}")
        if psnr_series:
            title_parts.append(f"PSNR: {psnr_series[-1]:.2f} dB")
        if ssim_series:
            title_parts.append(f"SSIM: {ssim_series[-1]:.4f}")
        if sam_series:
            title_parts.append(f"SAM: {sam_series[-1]:.4f} rad")
        fig.suptitle(" | ".join(title_parts), fontsize=15, fontweight='bold')

        # -------------------------------------------------------------
        # ROW 1: Spatial & Spectral Reconstruction Images
        # -------------------------------------------------------------
        # [0, 0] CASSI 2D Measurement
        axs[0, 0].imshow(meas_2d, cmap='gray')
        axs[0, 0].set_title(f"CASSI 2D Measurement\nShape: {meas_2d.shape}", fontsize=11)
        axs[0, 0].axis('off')

        # [0, 1] Ground Truth Band 15
        axs[0, 1].imshow(gt_band, cmap='gray', vmin=0.0, vmax=1.0)
        axs[0, 1].set_title(f"Ground Truth (Band {band_idx} ~550 nm)\nShape: {gt_band.shape}", fontsize=11)
        axs[0, 1].axis('off')

        # [0, 2] Model Prediction Band 15
        axs[0, 2].imshow(pred_band, cmap='gray', vmin=0.0, vmax=1.0)
        axs[0, 2].set_title(f"Prediction (Band {band_idx} ~550 nm)\nShape: {pred_band.shape}", fontsize=11)
        axs[0, 2].axis('off')

        # [0, 3] Absolute Error Map
        im_err = axs[0, 3].imshow(err_map, cmap='inferno')
        axs[0, 3].set_title(f"Absolute Error Map (|Pred - GT|)\nMax Error: {np.max(err_map):.4f}", fontsize=11)
        fig.colorbar(im_err, ax=axs[0, 3], fraction=0.046, pad=0.04)
        axs[0, 3].axis('off')

        # -------------------------------------------------------------
        # ROW 2: Quantitative Convergence Curves
        # -------------------------------------------------------------
        # Helper to plot line or single point
        def _plot_metric(ax, series: List[float], color_style: str, point_style: str, title: str, ylabel: str):
            if len(series) > 1:
                x = list(range(1, len(series) + 1))
                ax.plot(x, series, color_style, linewidth=2.0)
            elif len(series) == 1:
                x = [epoch if epoch > 0 else 0]
                ax.plot(x, series, point_style, markersize=8)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)

        # [1, 0] Training Loss Curve
        _plot_metric(axs[1, 0], loss_series, 'r-', 'ro', "Training Loss Curve", "Physics Loss")

        # [1, 1] Validation PSNR (dB)
        _plot_metric(axs[1, 1], psnr_series, 'b-', 'bo', "Validation PSNR (dB)", "PSNR (dB)")

        # [1, 2] Validation SSIM
        _plot_metric(axs[1, 2], ssim_series, 'g-', 'go', "Validation SSIM", "SSIM")
        if ssim_series:
            axs[1, 2].set_ylim([0.0, 1.05])

        # [1, 3] Validation SAM (rad)
        _plot_metric(axs[1, 3], sam_series, 'm-', 'mo', "Validation SAM (radians)", "SAM (rad)")

        plt.tight_layout()
        plt.savefig(out_file, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return out_file


def generate_dashboard(
    epoch: int,
    cassi_meas: Union[torch.Tensor, np.ndarray],
    gt_cube: Union[torch.Tensor, np.ndarray],
    pred_cube: Union[torch.Tensor, np.ndarray],
    history: Optional[Dict[str, List[Any]]] = None,
    output_dir: Optional[str] = None,
    scene_name: Optional[str] = None,
    band_idx: int = 15,
    save_path: Optional[str] = None,
) -> str:
    """
    Functional interface to render and save the 2x4 visual dashboard.
    
    Args:
        epoch: Current training epoch index.
        cassi_meas: CASSI 2D measurement.
        gt_cube: Ground truth HSI cube.
        pred_cube: Model predicted HSI cube.
        history: Metric curves dictionary.
        output_dir: Optional dashboard output directory.
        scene_name: Optional scene identifier string.
        band_idx: Spectral band index (default 15).
        save_path: Optional explicit output path.
        
    Returns:
        Absolute filesystem path of saved PNG file.
    """
    generator = VisualDashboardGenerator(output_dir=output_dir)
    return generator.generate(
        epoch=epoch,
        cassi_meas=cassi_meas,
        gt_cube=gt_cube,
        pred_cube=pred_cube,
        history=history,
        scene_name=scene_name,
        band_idx=band_idx,
        save_path=save_path,
    )


if __name__ == '__main__':
    # Standalone unit self-test
    print("Testing VisualDashboardGenerator & generate_dashboard...")
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = VisualDashboardGenerator(output_dir=tmpdir)
        dummy_meas = np.random.rand(256, 286).astype(np.float32)
        dummy_gt = np.random.rand(31, 256, 256).astype(np.float32)
        dummy_pred = np.random.rand(31, 256, 256).astype(np.float32)
        dummy_history = {
            'loss': [0.15, 0.12, 0.08],
            'psnr': [25.0, 28.5, 31.2],
            'ssim': [0.70, 0.82, 0.88],
            'sam': [0.65, 0.45, 0.35],
        }

        out_path = gen.generate(
            epoch=10,
            cassi_meas=dummy_meas,
            gt_cube=dummy_gt,
            pred_cube=dummy_pred,
            history=dummy_history,
            scene_name="synthetic_scene_01"
        )
        assert os.path.exists(out_path), f"Dashboard file not created: {out_path}"
        assert os.path.getsize(out_path) > 1000, f"Dashboard file too small: {os.path.getsize(out_path)} bytes"
        print(f"[+] Multi-epoch dashboard generated successfully: {out_path} ({os.path.getsize(out_path)} bytes)")

        # Single-epoch / dry-run test (epoch 0)
        single_path = gen.generate(
            epoch=0,
            cassi_meas=dummy_meas,
            gt_cube=dummy_gt,
            pred_cube=dummy_pred,
            history={'loss': [0.10], 'psnr': [27.0], 'ssim': [0.75], 'sam': [0.5]},
            save_path=os.path.join(tmpdir, "dashboard_ep0000.png")
        )
        assert os.path.exists(single_path)
        assert os.path.getsize(single_path) > 1000
        print(f"[+] Single-epoch (dry-run) dashboard generated successfully: {single_path} ({os.path.getsize(single_path)} bytes)")

    print("[+] All VisualDashboardGenerator self-tests passed successfully.")
