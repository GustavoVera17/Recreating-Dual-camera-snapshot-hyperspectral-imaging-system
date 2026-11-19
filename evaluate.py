"""
Comprehensive Evaluation of Spectral Image Reconstruction CNN (Xie_Rep_2)
across all 10 CAVE Test Scenes.
Supports:
- Strategy (i): Untrained per-scene self-supervised optimization (Table 1 of Xie et al., 2022)
- Strategy (ii): Pre-trained feed-forward inference (Table 2 of Xie et al., 2022)
- Strategy (ii) + Closed-Loop Fine-Tuning (Tables 3-5 of Xie et al., 2022)
Evaluates global PSNR, per-band mean PSNR, SSIM, SAM (deg & rad), dead bands, and runtime.
"""

import os
import sys
import csv
import time
import argparse
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.sir_cnn import SpectralImageReconstructionCNN
from models.forward_model import DualCameraForwardModel
from losses.physics_loss import PhysicsLoss
from utils.dataset import CAVEDualDataset
from utils.metrics import evaluate_cube


def _resolve_weights_path(weights_path: Optional[str]) -> Optional[str]:
    """Resolves checkpoint path with fallback to compact FP16 repository weights."""
    if weights_path is not None:
        return weights_path
    candidates = [
        os.path.join(BASE_DIR, "results", "checkpoints", "best_model.pth"),
        os.path.join(BASE_DIR, "results", "checkpoints", "best_model_fp16.pth"),
        os.path.join(BASE_DIR, "results", "weights", "best_model_fp16.pth"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def _load_weights_into_model(model: nn.Module, weights_path: str, device: torch.device) -> None:
    """Loads FP32/FP16 checkpoint strictly allowing only known module alias keys."""
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    state_dict = {k: v.float() if (torch.is_tensor(v) and v.dtype == torch.float16) else v for k, v in state_dict.items()}
    incompat = model.load_state_dict(state_dict, strict=False)
    alias_prefixes = ('stage1_ascension.', 'dimensional_ascension.', 'stage2.')
    non_alias_missing = [k for k in incompat.missing_keys if not k.startswith(alias_prefixes)]
    if non_alias_missing or incompat.unexpected_keys:
        raise RuntimeError(
            f"Incompatible checkpoint {weights_path}: missing={non_alias_missing}, unexpected={incompat.unexpected_keys}"
        )


def _calibrate_bn_single_scene(model: nn.Module, meas_cassi: torch.Tensor) -> None:
    """Synchronizes BatchNorm running stats cleanly for untrained single-scene Strategy (i) before eval()."""
    saved_momenta = []
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            saved_momenta.append((mod, mod.momentum))
            mod.momentum = 1.0
    model.train()
    with torch.no_grad():
        _ = model(meas_cassi)
    for mod, m in saved_momenta:
        mod.momentum = m
    model.eval()


def evaluate(
    weights_path: Optional[str] = None,
    hrnet_type: str = 'w32',
    device_str: str = 'auto',
    patch_size: int = 256,
    full_scene: bool = False,
    refine_steps: int = 0,
    refine_lr: float = 2e-4,
    refine_scheduler: str = 'cosine',
    refine_decay_step: int = 500,
    refine_decay_gamma: float = 0.2,
    untrained: bool = False,
    beta_spec: float = 0.0,
    beta_tv: float = 0.0,
    output_csv: Optional[str] = None,
    save_cubes_dir: Optional[str] = None,
    no_readme_update: bool = False,
    no_checkpointing: bool = False,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str == 'auto' else torch.device(device_str)
    use_cp = not bool(no_checkpointing)
    resolved_weights = None if untrained else _resolve_weights_path(weights_path)

    mode_str = (
        "Strategy (i): Untrained Self-Supervised" if untrained
        else ("Strategy (ii) + Closed-Loop Fine-Tune" if refine_steps > 0 else "Strategy (ii): Feed-Forward")
    )
    print("=" * 84)
    print(f"[*] Evaluating Xie_Rep_2 (SIR-CNN HRNet-{hrnet_type.upper()}) on 10 CAVE Test Scenes")
    print(f"    Mode         : {mode_str}")
    print(f"    Device       : {device}")
    print(f"    Weights Path : {'None (Untrained random init per scene)' if untrained else resolved_weights}")
    print(f"    Eval Geometry: {'Full 512x512' if full_scene else f'{patch_size}x{patch_size} Patch'}")
    print(f"    Refine Steps : {refine_steps} (lr={refine_lr}, sched={refine_scheduler}, beta_spec={beta_spec}, beta_tv={beta_tv})")
    print("=" * 84)

    dataset_test = CAVEDualDataset(
        split='test',
        patch_size=None if full_scene else patch_size,
        full_scene=full_scene,
        preload=True,
        synthetic_fallback=True,
    )
    num_scenes = len(dataset_test)

    model = SpectralImageReconstructionCNN(
        in_channels=1,
        num_bands=31,
        hrnet_type=hrnet_type,
        use_checkpoint=use_cp,
    ).to(device)

    base_state_dict = None
    if not untrained:
        if resolved_weights is not None and os.path.exists(resolved_weights):
            _load_weights_into_model(model, resolved_weights, device)
            base_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print("[+] Model weights loaded successfully.")
        else:
            base_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print("[!] Warning: Weights file not found. Evaluating with initialized weights.")

    forward_model = DualCameraForwardModel().to(device)
    criterion = PhysicsLoss(forward_model=forward_model, beta_spec=beta_spec, beta_tv=beta_tv).to(device)

    if save_cubes_dir is not None:
        os.makedirs(save_cubes_dir, exist_ok=True)

    results = []
    psnr_list, psnr_band_list, ssim_list, sam_rad_list, sam_deg_list, time_list = [], [], [], [], [], []
    total_dead_bands = 0

    for i in range(num_scenes):
        sample = dataset_test[i]
        meas_cassi = sample['meas_cassi'].unsqueeze(0).to(device)
        meas_rgb = sample['meas_rgb'].unsqueeze(0).to(device)
        mask = sample['mask'].unsqueeze(0).to(device)
        gt_cube = sample['cube']
        scene_name = sample.get('scene_name', f"scene_{i+1:02d}")

        t_scene_start = time.time()
        if untrained:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = SpectralImageReconstructionCNN(
                in_channels=1,
                num_bands=31,
                hrnet_type=hrnet_type,
                use_checkpoint=use_cp,
            ).to(device)

        if refine_steps > 0:
            if not untrained:
                model.load_state_dict(base_state_dict, strict=True)
                # Freeze BatchNorm running mean/var across batch_size=1 refinement while updating weights
                model.eval()
            else:
                model.train()

            opt = torch.optim.Adam(model.parameters(), lr=refine_lr)
            sched = None
            if refine_scheduler == 'step':
                sched = torch.optim.lr_scheduler.StepLR(opt, step_size=refine_decay_step, gamma=refine_decay_gamma)
            elif refine_scheduler == 'cosine':
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=refine_steps, eta_min=max(1e-6, refine_lr * 0.05))

            for _ in range(refine_steps):
                opt.zero_grad()
                pred_step = model(meas_cassi)
                loss_res = criterion(pred_step.float(), meas_cassi.float(), meas_rgb.float(), mask=mask.float())
                loss_res.loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
                if sched is not None:
                    sched.step()

            if untrained:
                _calibrate_bn_single_scene(model, meas_cassi)
        else:
            model.eval()

        model.eval()
        with torch.no_grad():
            pred = model(meas_cassi)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        scene_time_s = time.time() - t_scene_start

        pred_cube = pred[0].float().cpu()
        band_means = pred_cube.mean(dim=(1, 2)).numpy()
        scene_dead = int((band_means < 1e-3).sum())
        total_dead_bands += scene_dead

        if save_cubes_dir is not None:
            np.save(os.path.join(save_cubes_dir, f"{scene_name}.npy"), pred_cube.numpy())

        metrics = evaluate_cube(pred_cube, gt_cube)
        psnr_v = float(metrics['psnr'])
        psnr_band_v = float(metrics['psnr_band'])
        ssim_v = float(metrics['ssim'])
        sam_rad = float(metrics['sam_rad'])
        sam_deg = float(metrics['sam_deg'])

        psnr_list.append(psnr_v)
        psnr_band_list.append(psnr_band_v)
        ssim_list.append(ssim_v)
        sam_rad_list.append(sam_rad)
        sam_deg_list.append(sam_deg)
        time_list.append(scene_time_s)

        results.append([
            scene_name, f"{psnr_v:.2f}", f"{psnr_band_v:.2f}",
            f"{ssim_v:.4f}", f"{sam_deg:.2f}", f"{sam_rad:.4f}", f"{scene_time_s:.2f}",
        ])
        print(
            f"  [{i+1:02d}/{num_scenes}] {scene_name:<26} | PSNR: {psnr_v:5.2f} dB (band {psnr_band_v:5.2f}) "
            f"| SSIM: {ssim_v:.4f} | SAM: {sam_deg:5.2f} deg | Dead: {scene_dead} | {scene_time_s:6.2f}s"
        )

    mean_psnr = float(np.mean(psnr_list))
    mean_psnr_band = float(np.mean(psnr_band_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_sam_deg = float(np.mean(sam_deg_list))
    mean_sam_rad = float(np.mean(sam_rad_list))
    mean_time = float(np.mean(time_list))

    print("\n" + "=" * 92)
    print("                     SUMMARY TABLE: REPLICATION VS PAPER (CAVE)")
    print("=" * 92)
    header = f"{'Method':<35} | {'Global PSNR':<11} | {'Band PSNR':<10} | {'SSIM':<8} | {'SAM (deg)':<10} | {'Time/Scene':<10}"
    print(header)
    print("-" * len(header))
    print(f"{'Paper Table 1 (Strategy i)':<35} | {'43.88':<11} | {'--':<10} | {'0.9610':<8} | {'9.05°':<10} | {'~300s':<10}")
    print(f"{'Paper Table 2 (Strategy ii)':<35} | {'39.15':<11} | {'--':<10} | {'0.9430':<8} | {'11.10°':<10} | {'Real-time':<10}")
    print(f"{mode_str:<35} | {mean_psnr:<11.2f} | {mean_psnr_band:<10.2f} | {mean_ssim:<8.4f} | {mean_sam_deg:<10.2f} | {mean_time:<9.2f}s")
    print("=" * 92)
    print(f"[+] Total dead bands (<1e-3 mean) across {num_scenes} scenes x 31 bands: {total_dead_bands}")

    csv_path = os.path.abspath(output_csv) if output_csv else os.path.join(BASE_DIR, "results", "test_evaluation_10scenes.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_name", "psnr_db", "psnr_band_db", "ssim", "sam_deg", "sam_rad", "time_s"])
        for row in results:
            writer.writerow(row)
        writer.writerow([
            "MEAN", f"{mean_psnr:.2f}", f"{mean_psnr_band:.2f}",
            f"{mean_ssim:.4f}", f"{mean_sam_deg:.2f}", f"{mean_sam_rad:.4f}", f"{mean_time:.2f}",
        ])
    print(f"[+] Per-scene test results saved to: {csv_path}")

    if not no_readme_update:
        readme_path = os.path.join(BASE_DIR, "README.md")
        if os.path.exists(readme_path):
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_content = f.read()
                placeholder = "| **Xie_Rep_2 (This Repo)** | *Training in Progress* | *Training in Progress* | *Training in Progress* | **118.2 MB** |"
                rep_row = f"| **Xie_Rep_2 (This Repo)** | **{mean_psnr:.2f}** | **{mean_ssim:.4f}** | **{mean_sam_deg:.2f}°** | **118.2 MB** |"
                if placeholder in readme_content:
                    with open(readme_path, "w", encoding="utf-8") as f:
                        f.write(readme_content.replace(placeholder, rep_row))
                    print(f"[+] Updated benchmark results in {readme_path}")
            except Exception as e:
                print(f"[!] Warning: Could not update README.md: {e}")

    return {
        'mean_psnr': mean_psnr,
        'mean_psnr_band': mean_psnr_band,
        'mean_ssim': mean_ssim,
        'mean_sam_deg': mean_sam_deg,
        'mean_sam_rad': mean_sam_rad,
        'mean_time_s': mean_time,
        'total_dead_bands': total_dead_bands,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Xie_Rep_2 Model on 10 CAVE Test Scenes")
    parser.add_argument('--weights', type=str, default=None, help="Path to checkpoint weights")
    parser.add_argument('--hrnet-type', type=str, default='w32', choices=['w18', 'w32', 'w48'])
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--patch-size', type=int, default=256, help="Spatial patch size matching training")
    parser.add_argument('--full-scene', action='store_true', help="Evaluate on uncropped 512x512 scene")
    parser.add_argument('--untrained', action='store_true', help="Use untrained fresh init per scene (Strategy i)")
    parser.add_argument('--refine-steps', type=int, default=0, help="Self-supervised dual-camera optimization steps")
    parser.add_argument('--refine-lr', type=float, default=2e-4, help="Learning rate for closed-loop optimization")
    parser.add_argument('--refine-scheduler', type=str, default='cosine', choices=['none', 'step', 'cosine'])
    parser.add_argument('--refine-decay-step', type=int, default=500, help="StepLR decay interval during refinement")
    parser.add_argument('--refine-decay-gamma', type=float, default=0.2, help="StepLR gamma during refinement")
    parser.add_argument('--beta-spec', type=float, default=0.0, help="Spectral continuity prior weight")
    parser.add_argument('--beta-tv', type=float, default=0.0, help="Spatial TV prior weight")
    parser.add_argument('--output-csv', type=str, default=None, help="Custom output CSV path")
    parser.add_argument('--save-cubes-dir', type=str, default=None, help="Optional dir to save reconstructed .npy cubes")
    parser.add_argument('--no-readme-update', action='store_true', help="Skip updating README.md")
    parser.add_argument('--no-checkpointing', action='store_true', help="Disable HRNet gradient checkpointing")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for --untrained initialization")
    args = parser.parse_args()

    evaluate(
        weights_path=args.weights,
        hrnet_type=args.hrnet_type,
        device_str=args.device,
        patch_size=args.patch_size,
        full_scene=args.full_scene,
        refine_steps=args.refine_steps,
        refine_lr=args.refine_lr,
        refine_scheduler=args.refine_scheduler,
        refine_decay_step=args.refine_decay_step,
        refine_decay_gamma=args.refine_decay_gamma,
        untrained=args.untrained,
        beta_spec=args.beta_spec,
        beta_tv=args.beta_tv,
        output_csv=args.output_csv,
        save_cubes_dir=args.save_cubes_dir,
        no_readme_update=args.no_readme_update,
        no_checkpointing=args.no_checkpointing,
        seed=args.seed,
    )
