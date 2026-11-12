"""
Training Pipeline for Dual-Camera Snapshot Hyperspectral Imaging (Xie_Rep_2).

This script implements the complete training harness described in Xie et al. (2022),
"Dual camera snapshot hyperspectral imaging system via physics-informed learning", Optics Express,
adhering to the Yamawaki_Rep_2 engineering standards:

Key Architectural Highlights:
1. Pure Self-Supervised Physics-Informed Training:
   No paired ground-truth hyperspectral cubes are used to compute the optimization loss:
       Loss_total = Loss_grayscale + alpha * Loss_color (MSE)
2. PyTorch Modern AMP (`torch.amp`):
   FP16 automatic mixed precision via `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')`
   with seamless CPU fallback.
3. Adam Optimizer & StepLR Scheduling:
   Adam (lr=0.001, betas=(0.9, 0.999), eps=1e-8) with StepLR decay (gamma=0.2 every 300 epochs)
   across 1200 total epochs.
4. Periodic Visual Dashboards:
   Headless Matplotlib 2x4 diagnostic figures saved every `dash_interval` epochs
   to 'results/dashboards/dashboard_ep{epoch:04d}.png'.
5. Structured CSV Logging:
   Tracks 8 standardized columns with dynamic rolling average ETA formatted strictly as 'DD:HH:MM:SS'.
6. Checkpointing:
   Persists best validation model to 'results/checkpoints/best_model.pth' and
   latest state to 'results/checkpoints/latest_checkpoint.pth'.
7. Requirement R4 Verification Mode (`--dry-run`):
   Executes 1 forward pass, 1 backward pass with GradScaler, 1 validation evaluation,
   generates 'results/dashboards/dashboard_ep0000.png', writes 1 row to 'results/training_log.csv',
   and exits cleanly with code 0 WITHOUT launching training loops.
"""

import os
import sys
import time
import argparse
from typing import Optional, Dict, Any, Tuple
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.sir_cnn import SpectralImageReconstructionCNN
from models.forward_model import DualCameraForwardModel
from losses.physics_loss import PhysicsLoss
from utils.dataset import CAVEDualDataset, worker_init_fn
from utils.metrics import evaluate_cube, calc_psnr, calc_ssim, calc_sam
from utils.logger import CSVLogger, format_eta
from utils.dashboard import VisualDashboardGenerator


def parse_args():
    """Parses command-line arguments for training and dry-run execution."""
    parser = argparse.ArgumentParser(
        description="Xie et al. (2022) Physics-Informed Dual-Camera HSI Training Harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core training hyperparameters per Xie et al. (2022) Section 3.1
    parser.add_argument(
        '--epochs',
        type=int,
        default=1200,
        help="Total training epochs per Xie et al. (2022)"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help="Initial Adam learning rate"
    )
    parser.add_argument(
        '--batch-size', '--batch_size',
        type=int,
        default=16,
        dest='batch_size',
        help="Training batch size"
    )
    parser.add_argument(
        '--patch-size', '--patch_size',
        type=int,
        default=256,
        dest='patch_size',
        help="Spatial patch crop dimension (HxW)"
    )
    parser.add_argument(
        '--decay-step', '--decay_step',
        type=int,
        default=300,
        dest='decay_step',
        help="StepLR decay interval in epochs"
    )
    parser.add_argument(
        '--decay-gamma', '--decay_gamma',
        type=float,
        default=0.2,
        dest='decay_gamma',
        help="StepLR multiplicative decay factor"
    )
    parser.add_argument(
        '--dash-interval', '--dash_interval',
        type=int,
        default=10,
        dest='dash_interval',
        help="Epoch interval for rendering visual PNG dashboards"
    )
    parser.add_argument(
        '--hrnet-type', '--hrnet_type',
        type=str,
        default='w32',
        choices=['w18', 'w32', 'w48'],
        dest='hrnet_type',
        help="HRNet backbone capacity configuration"
    )
    parser.add_argument(
        '--dry-run', '--dry_run',
        action='store_true',
        dest='dry_run',
        help="Execute 1 mini-batch dry-run verification and exit without training (Requirement R4)"
    )

    # Dataset & directory paths
    parser.add_argument(
        '--data-dir', '--data_dir', '--root-dir',
        type=str,
        default=None,
        dest='data_dir',
        help="Path to CAVE dataset directory (auto-resolves if None)"
    )
    parser.add_argument(
        '--mask-path', '--mask_path',
        type=str,
        default=None,
        dest='mask_path',
        help="Path to 512x512 coded aperture mask npy file"
    )
    parser.add_argument(
        '--results-dir', '--results_dir',
        type=str,
        default=None,
        dest='results_dir',
        help="Directory to store CSV logs, checkpoints, and dashboards"
    )
    parser.add_argument(
        '--num-workers', '--num_workers',
        type=int,
        default=0,
        dest='num_workers',
        help="DataLoader worker processes"
    )
    parser.add_argument(
        '--no-checkpointing',
        action='store_true',
        dest='no_checkpointing',
        help="Disable gradient checkpointing in HRNet backbone"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help="Compute device selection"
    )
    parser.add_argument(
        '--beta-spec', '--beta_spec',
        type=float,
        default=0.5,
        dest='beta_spec',
        help="Self-supervised spectral continuity regularization weight"
    )
    parser.add_argument(
        '--beta-tv', '--beta_tv',
        type=float,
        default=0.05,
        dest='beta_tv',
        help="Self-supervised spatial total variation regularization weight"
    )
    parser.add_argument(
        '--log-filename', '--log_filename',
        type=str,
        default='training_log.csv',
        dest='log_filename',
        help="CSV log filename inside results_dir"
    )
    parser.add_argument(
        '--dashboards-dir', '--dashboards_dir',
        type=str,
        default=None,
        dest='dashboards_dir',
        help="Optional override directory for visual dashboards"
    )
    parser.add_argument(
        '--git-sync', '--git_sync',
        action='store_true',
        dest='git_sync',
        help="Enable periodic git commit/push during training"
    )
    parser.add_argument(
        '--git-sync-interval', '--git_sync_interval',
        type=int,
        default=50,
        dest='git_sync_interval',
        help="Epoch interval for periodic git commit/push of logs & dashboards"
    )
    parser.add_argument(
        '--weights-sync-interval', '--weights_sync_interval',
        type=int,
        default=100,
        dest='weights_sync_interval',
        help="Epoch interval for periodic git commit/push of compact FP16 weights (0 disables)"
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help="Overwrite existing CSV log file instead of appending"
    )

    return parser.parse_args()


def _dedup_fp16_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Serializes unique tensor storages in FP16 so checkpoints stay under GitHub's 100MB limit."""
    seen_ptrs = set()
    compact_state: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        compact_state[k] = v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
    return compact_state


def _resolve_amp_config(device: torch.device) -> Tuple[Optional[torch.dtype], Optional[torch.amp.GradScaler]]:
    """Selects BF16 autocast without GradScaler on supported GPUs, falling back to FP16+GradScaler."""
    if device.type != 'cuda' or not torch.cuda.is_available():
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler('cuda')


def _execute_train_step(
    model: nn.Module,
    criterion: PhysicsLoss,
    optimizer: torch.optim.Optimizer,
    meas_cassi: torch.Tensor,
    meas_rgb: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    scaler: Optional[torch.amp.GradScaler],
    max_grad_norm: float = 5.0,
):
    """Runs one forward/backward optimization step with AMP forward and FP32 physics loss."""
    optimizer.zero_grad()
    amp_ctx = (
        torch.amp.autocast('cuda', dtype=amp_dtype)
        if (device.type == 'cuda' and amp_dtype is not None)
        else nullcontext()
    )
    with amp_ctx:
        pred_cube = model(meas_cassi)

    # Keep forward physics operators (31-band summation & CRF dot product) in FP32 to avoid half-precision overflow.
    loss_res = criterion(pred_cube.float(), meas_cassi.float(), meas_rgb.float(), mask=mask.float())
    loss_total = loss_res.loss_total

    if scaler is not None:
        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

    return loss_res


def run_dry_run_verification(
    model: nn.Module,
    forward_model: DualCameraForwardModel,
    criterion: PhysicsLoss,
    optimizer: torch.optim.Optimizer,
    amp_dtype: Optional[torch.dtype],
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    train_dataset: CAVEDualDataset,
    val_dataset: CAVEDualDataset,
    logger: CSVLogger,
    dashboard_gen: VisualDashboardGenerator,
    dashboards_dir: str,
    checkpoints_dir: str,
    args: argparse.Namespace,
) -> None:
    """
    Executes non-training dry-run verification satisfying Requirement R4:
    1. Runs 1 mini-batch forward pass with PyTorch AMP.
    2. Runs 1 backward pass with GradScaler and verifies gradient backpropagation.
    3. Runs 1 validation evaluation and computes PSNR, SSIM, SAM (radians).
    4. Appends 1 row to results/training_log.csv with DD:HH:MM:SS ETA.
    5. Generates test visual dashboard results/dashboards/dashboard_ep0000.png.
    6. Exits cleanly with code 0 without launching epoch training loops.
    """
    print("\n" + "=" * 78)
    print("  [+] INITIATING REQUIREMENT R4 DRY-RUN VERIFICATION (NON-TRAINING MODE)")
    print("=" * 78)

    # 1. Prepare mini-batch
    batch_size = min(args.batch_size, 2)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )
    batch = next(iter(loader))

    meas_cassi = batch['meas_cassi'].to(device)  # [B, 1, H, W + 30]
    meas_rgb = batch['meas_rgb'].to(device)      # [B, 3, H, W]
    mask = batch['mask'].to(device)              # [B, H, W]
    gt_cube = batch['cube'].to(device)           # [B, 31, H, W]

    print(f"[*] Input Batch Shapes:")
    print(f"    - meas_cassi : {tuple(meas_cassi.shape)}")
    print(f"    - meas_rgb   : {tuple(meas_rgb.shape)}")
    print(f"    - mask       : {tuple(mask.shape)}")
    print(f"    - gt_cube    : {tuple(gt_cube.shape)}")

    # 2. Forward & backward pass with PyTorch AMP (BF16 preferred, FP32 physics loss)
    model.train()
    t_start = time.time()
    loss_res = _execute_train_step(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        meas_cassi=meas_cassi,
        meas_rgb=meas_rgb,
        mask=mask,
        device=device,
        amp_dtype=amp_dtype,
        scaler=scaler,
    )
    loss_total = loss_res.loss_total

    forward_backward_time = time.time() - t_start

    # Verify gradients exist across trainable parameters
    active_grads = [
        p.grad for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if len(active_grads) == 0:
        raise RuntimeError("Dry-run failed: No gradients accumulated on model parameters!")

    has_nan = any(torch.isnan(g).any().item() or torch.isinf(g).any().item() for g in active_grads)
    if has_nan:
        raise RuntimeError("Dry-run failed: NaN or Inf detected in accumulated gradients!")

    print(f"[+] Mini-batch forward & backward pass completed in {forward_backward_time:.3f}s.")
    print(f"    - AMP dtype      : {amp_dtype if amp_dtype is not None else 'float32'}")
    print(f"    - Loss Total     : {loss_total.item():.6f}")
    print(f"    - Loss CASSI     : {loss_res.loss_cassi.item():.6f}")
    print(f"    - Loss RGB       : {loss_res.loss_rgb.item():.6f}")
    print(f"    - Gradients OK   : {len(active_grads)} parameter tensors updated.")

    # 4. Mini validation evaluation
    model.eval()
    val_sample = val_dataset[0]
    val_cassi = val_sample['meas_cassi'].unsqueeze(0).to(device)
    val_gt = val_sample['cube']  # [31, H, W]

    amp_ctx = (
        torch.amp.autocast('cuda', dtype=amp_dtype)
        if (device.type == 'cuda' and amp_dtype is not None)
        else nullcontext()
    )
    with torch.no_grad():
        with amp_ctx:
            val_pred = model(val_cassi)

    val_pred_cube = val_pred[0].float().cpu()  # [31, H, W]
    metrics = evaluate_cube(val_pred_cube, val_gt)

    val_psnr = float(metrics['psnr'])
    val_ssim = float(metrics['ssim'])
    val_sam = float(metrics['sam_rad'])

    print(f"[+] Validation evaluation completed:")
    print(f"    - Val PSNR (dB)  : {val_psnr:.2f} dB")
    print(f"    - Val SSIM       : {val_ssim:.4f}")
    print(f"    - Val SAM (rad)  : {val_sam:.4f} rad")

    # 5. Write 1 row to results/training_log.csv with DD:HH:MM:SS ETA
    log_row = logger.log_epoch(
        epoch=0,
        train_loss=loss_total.item(),
        val_psnr=val_psnr,
        val_ssim=val_ssim,
        val_sam=val_sam,
        lr=args.lr,
        epoch_time=forward_backward_time,
        total_epochs=args.epochs,
    )
    print(f"[+] Progress logged to CSV ({logger.log_path}):")
    print(f"    {log_row}")

    # 6. Generate test dashboard results/dashboards/dashboard_ep0000.png
    dash_path = os.path.join(dashboards_dir, "dashboard_ep0000.png")
    saved_dash = dashboard_gen.generate(
        epoch=0,
        cassi_meas=val_cassi[0].cpu(),
        gt_cube=val_gt,
        pred_cube=val_pred_cube,
        history={
            'loss': [loss_total.item()],
            'psnr': [val_psnr],
            'ssim': [val_ssim],
            'sam': [val_sam],
        },
        scene_name=val_sample.get('scene_name', 'dry_run_scene'),
        band_idx=15,
        save_path=dash_path,
    )
    if not os.path.exists(saved_dash) or os.path.getsize(saved_dash) == 0:
        raise RuntimeError(f"Dry-run dashboard was not successfully written to {saved_dash}")

    print(f"[+] Visual dashboard generated: {saved_dash} ({os.path.getsize(saved_dash)} bytes)")

    # Ensure checkpoints directory exists
    os.makedirs(checkpoints_dir, exist_ok=True)

    print("=" * 78)
    print("[+] DRY RUN VERIFICATION PASSED SUCCESSFULLY.")
    print("=" * 78)
    sys.exit(0)


def main():
    args = parse_args()

    # 1. Device Resolution
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    device_name = torch.cuda.get_device_name(0) if (device.type == 'cuda' and torch.cuda.is_available()) else "CPU"

    # 2. Directory Layout Setup
    if args.results_dir is not None:
        results_dir = os.path.abspath(args.results_dir)
    else:
        results_dir = os.path.join(BASE_DIR, "results")

    dashboards_dir = (
        os.path.abspath(args.dashboards_dir)
        if args.dashboards_dir is not None
        else os.path.join(results_dir, "dashboards")
    )
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    weights_sync_dir = os.path.join(results_dir, "weights")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(dashboards_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(weights_sync_dir, exist_ok=True)

    # 3. CSV Logger and Visual Dashboard Generator Initialization
    csv_path = os.path.join(results_dir, args.log_filename)
    logger = CSVLogger(log_path=csv_path, overwrite=args.fresh)
    dashboard_gen = VisualDashboardGenerator(output_dir=dashboards_dir)

    print("=" * 78)
    print("  Xie et al. (2022) Snapshot Hyperspectral Imaging Training Harness")
    print(f"  Device           : {device} ({device_name})")
    print(f"  HRNet Backbone   : HRNet-{args.hrnet_type.upper()}")
    print(f"  Batch / Patch    : {args.batch_size} / {args.patch_size}x{args.patch_size}")
    print(f"  Learning Rate    : {args.lr} (Decay: gamma={args.decay_gamma} every {args.decay_step} epochs)")
    print(f"  Total Epochs     : {args.epochs}")
    print(f"  Results Dir      : {results_dir}")
    print(f"  Dry-Run Mode     : {args.dry_run}")
    print("=" * 78)

    # 4. Dataset Loading
    use_checkpoint = not args.no_checkpointing
    train_dataset = CAVEDualDataset(
        root_dir=args.data_dir,
        split='train',
        patch_size=args.patch_size,
        is_train=True,
        mask_path=args.mask_path,
        synthetic_fallback=True,
        preload=True,
    )
    val_dataset = CAVEDualDataset(
        root_dir=args.data_dir,
        split='test',
        patch_size=args.patch_size,
        is_train=False,
        mask_path=args.mask_path,
        synthetic_fallback=True,
        preload=True,
    )

    # 5. Network Architecture, Optical Forward Model & Loss Objective
    model = SpectralImageReconstructionCNN(
        in_channels=1,
        num_bands=31,
        hrnet_type=args.hrnet_type,
        use_checkpoint=use_checkpoint,
    ).to(device)

    forward_model = DualCameraForwardModel().to(device)
    criterion = PhysicsLoss(
        forward_model=forward_model,
        beta_spec=args.beta_spec,
        beta_tv=args.beta_tv,
    ).to(device)

    # 6. Optimizer, Scheduler & Mixed Precision Scaler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.decay_step,
        gamma=args.decay_gamma,
    )

    amp_dtype, scaler = _resolve_amp_config(device)
    amp_label = str(amp_dtype).replace('torch.', '') if amp_dtype is not None else 'float32'
    print(f"[*] Precision mode   : {amp_label} forward (FP32 physics loss, scaler={'on' if scaler is not None else 'off'})")

    # 7. REQUIREMENT R4 CHECK: If dry-run, execute verification and exit immediately
    if args.dry_run:
        run_dry_run_verification(
            model=model,
            forward_model=forward_model,
            criterion=criterion,
            optimizer=optimizer,
            amp_dtype=amp_dtype,
            scaler=scaler,
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            logger=logger,
            dashboard_gen=dashboard_gen,
            dashboards_dir=dashboards_dir,
            checkpoints_dir=checkpoints_dir,
            args=args,
        )
        return

    # -------------------------------------------------------------------------
    # FULL PRODUCTION TRAINING LOOP (Authorized only post-audit in future phase)
    # -------------------------------------------------------------------------
    pin_mem = (device.type == 'cuda')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        worker_init_fn=worker_init_fn,
    )

    best_psnr = -1.0
    print(f"\n[*] Commencing full training cycle ({args.epochs} epochs)...")

    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            meas_cassi = batch['meas_cassi'].to(device)
            meas_rgb = batch['meas_rgb'].to(device)
            mask = batch['mask'].to(device)

            loss_res = _execute_train_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                meas_cassi=meas_cassi,
                meas_rgb=meas_rgb,
                mask=mask,
                device=device,
                amp_dtype=amp_dtype,
                scaler=scaler,
                max_grad_norm=5.0,
            )

            total_loss += loss_res.loss_total.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(1, num_batches)
        current_lr = float(optimizer.param_groups[0]['lr'])

        # Validation evaluation across all validation scenes
        model.eval()
        val_psnr_list, val_ssim_list, val_sam_list = [], [], []
        val_sample = val_dataset[0]
        val_cassi = val_sample['meas_cassi'].unsqueeze(0).to(device)
        val_gt = val_sample['cube']
        val_pred_cube = None

        amp_ctx = (
            torch.amp.autocast('cuda', dtype=amp_dtype)
            if (device.type == 'cuda' and amp_dtype is not None)
            else nullcontext()
        )
        with torch.no_grad():
            for v_idx in range(len(val_dataset)):
                v_item = val_dataset[v_idx]
                v_cassi = v_item['meas_cassi'].unsqueeze(0).to(device)
                v_gt = v_item['cube']
                with amp_ctx:
                    v_pred = model(v_cassi)
                v_pred_cpu = v_pred[0].float().cpu()
                if v_idx == 0:
                    val_pred_cube = v_pred_cpu
                v_metrics = evaluate_cube(v_pred_cpu, v_gt)
                val_psnr_list.append(float(v_metrics['psnr']))
                val_ssim_list.append(float(v_metrics['ssim']))
                val_sam_list.append(float(v_metrics['sam_rad']))

        val_psnr = float(np.mean(val_psnr_list))
        val_ssim = float(np.mean(val_ssim_list))
        val_sam = float(np.mean(val_sam_list))

        epoch_dur = time.time() - t_epoch_start

        # CSV Logging & ETA Calculation
        log_row = logger.log_epoch(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_psnr=val_psnr,
            val_ssim=val_ssim,
            val_sam=val_sam,
            lr=current_lr,
            epoch_time=epoch_dur,
            total_epochs=args.epochs,
        )

        eta_str = log_row['Estimated_Time_Remaining']
        print(
            f"[*] Ep {epoch:04d}/{args.epochs} | Loss: {avg_train_loss:.6f} | "
            f"PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | SAM: {val_sam:.4f} rad | "
            f"LR: {current_lr:.6f} | Time: {epoch_dur:.2f}s | ETA: {eta_str}"
        )

        # Checkpointing
        latest_ckpt_path = os.path.join(checkpoints_dir, "latest_checkpoint.pth")
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_sam': val_sam,
            },
            latest_ckpt_path,
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ckpt_path = os.path.join(checkpoints_dir, "best_model.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            # Save deduplicated FP16 copy (~60MB < 100MB GitHub limit)
            try:
                fp16_state = _dedup_fp16_state_dict(model)
                torch.save(fp16_state, os.path.join(checkpoints_dir, "best_model_fp16.pth"))
                torch.save(fp16_state, os.path.join(weights_sync_dir, "best_model_fp16.pth"))
            except Exception as e:
                print(f"  [!] FP16 checkpoint save notice: {e}")
            print(f"  [+] New Best PSNR: {best_psnr:.2f} dB (saved {best_ckpt_path})")

        # Visual Dashboard Generation
        if epoch % args.dash_interval == 0 or epoch == args.epochs:
            dashboard_gen.generate(
                epoch=epoch,
                cassi_meas=val_cassi[0].cpu(),
                gt_cube=val_gt,
                pred_cube=val_pred_cube,
                history=logger.get_history(),
                scene_name=val_sample.get('scene_name', 'val_scene'),
                band_idx=15,
            )

        # Periodic Git Push to GitHub when --git-sync is enabled
        sync_interval = max(1, int(args.git_sync_interval))
        weights_interval = int(args.weights_sync_interval)
        due_metrics = (epoch % sync_interval == 0 or epoch == args.epochs)
        due_weights = (weights_interval > 0 and (epoch % weights_interval == 0 or epoch == args.epochs))
        if args.git_sync and (due_metrics or due_weights):
            try:
                import subprocess
                targets = []
                if due_metrics:
                    for p in (csv_path, dashboards_dir, os.path.join(BASE_DIR, "README.md")):
                        if os.path.exists(p):
                            targets.append(os.path.relpath(p, BASE_DIR))
                if due_weights and os.path.exists(weights_sync_dir):
                    targets.append(os.path.relpath(weights_sync_dir, BASE_DIR))
                if targets:
                    subprocess.run(["git", "add", "--", *targets], cwd=BASE_DIR, check=True)
                    status_res = subprocess.run(
                        ["git", "status", "--porcelain", "--", *targets],
                        cwd=BASE_DIR,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    if status_res.stdout.strip():
                        tag = "metrics & weights" if due_weights else "metrics & dashboards"
                        subprocess.run(
                            ["git", "commit", "-m", f"Auto-sync: Training {tag} at epoch {epoch}"],
                            cwd=BASE_DIR,
                            check=True,
                        )
                        subprocess.run(["git", "push", "origin", "main"], cwd=BASE_DIR, check=True)
                        print(f"  [-] Synced epoch {epoch} ({tag}) to GitHub successfully.")
                    else:
                        print(f"  [-] No new git changes at epoch {epoch}.")
            except Exception as e:
                print(f"  [!] Git push notice at epoch {epoch}: {e}")


if __name__ == '__main__':
    main()
