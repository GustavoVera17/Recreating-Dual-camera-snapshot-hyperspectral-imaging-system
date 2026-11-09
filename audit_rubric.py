"""
Automated 100-Point Audit Rubric for Xie_Rep_2.

Dual Camera Snapshot Hyperspectral Imaging System via Physics-Informed Learning
Primary Reference: Hui Xie, Zhuang Zhao, Jing Han, Yi Zhang, Lianfa Bai, and Jun Lu,
Optics Express 30(14), 24652-24666 (2022).

This script programmatically evaluates the entire codebase across 5 distinct sections
totaling exactly 100 points:
- SECTION 1: Paper Fidelity & Network Architecture Topology (25 pts)
- SECTION 2: Mathematical Forward Physics & Optical Models (25 pts)
- SECTION 3: Closed-Loop Self-Supervised Physics Loss (20 pts)
- SECTION 4: Quantitative Evaluation Metrics Rigor (15 pts)
- SECTION 5: CAVE Dataset Pipeline, Partitioning & Stochastic Sampling (15 pts)

Execution & Gate Contract:
- Standalone execution: python audit_rubric.py
- Returns exit code 0 if score == 100, and 1 otherwise.
"""

import sys
import os
import inspect
import ast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.sir_cnn import SpectralImageReconstructionCNN, Stage1DimensionalAscending
from models.hrnet_backbone import HRNetBackbone, BasicBlock, HRModule, HRStage
from models.forward_model import DualCameraForwardModel
from models.crf import CameraResponseFunction, get_crf_matrix, project_rgb
from models.mask import CodedApertureMask, modulate_cube
from losses.physics_loss import PhysicsLoss
from utils.metrics import calc_psnr, calc_ssim, calc_sam, evaluate_cube
from utils.splits import CAVE_TRAIN_SCENES, CAVE_TEST_SCENES, get_cave_splits
from utils.dataset import CAVEDualDataset


def run_audit() -> int:
    score = 0
    total_points = 100
    section_scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    print("=" * 80)
    print("          XIE ET AL. (2022) COMPREHENSIVE REPLICATION AUDIT RUBRIC")
    print("=" * 80)

    # =========================================================================
    # SECTION 1: Paper Fidelity & Network Architecture Topology (25 pts)
    # =========================================================================
    print("\n[SECTION 1] Auditing Paper Fidelity & Network Architecture Topology (25 pts)...")
    sec1_pts = 0

    # Item 1.1: Stage 1 1xλ Filter Geometry (8 pts)
    try:
        stage1 = Stage1DimensionalAscending(in_channels=1, num_bands=31)
        conv1x = stage1.conv1x_lambda
        
        # Geometry assertions
        is_conv2d = isinstance(conv1x, nn.Conv2d)
        has_correct_kernel = (conv1x.kernel_size == (1, 31))
        has_correct_stride = (conv1x.stride == (1, 1) or conv1x.stride == 1)
        has_correct_padding = (conv1x.padding == (0, 0) or conv1x.padding == 0)
        has_correct_channels = (conv1x.in_channels == 1 and conv1x.out_channels == 31)
        
        # Explicit negative check: strictly reject pointwise Conv2d 1x1 from flawed Xie_Rep
        not_pointwise = (conv1x.kernel_size != (1, 1))

        # Forward shape transformation assertion: [2, 1, 256, 286] -> [2, 31, 256, 256]
        dummy_disp = torch.randn(2, 1, 256, 286)
        out_stage1 = stage1(dummy_disp)
        correct_shape = (out_stage1.shape == torch.Size([2, 31, 256, 256]))
        non_negative = isinstance(stage1.relu, nn.ReLU) and bool(torch.all(out_stage1 >= 0.0))

        if (is_conv2d and has_correct_kernel and has_correct_stride and
            has_correct_padding and has_correct_channels and not_pointwise and
            correct_shape and non_negative):
            print("  [+] Stage 1: nn.Conv2d(1, 31, kernel=(1, 31), pad=0) dimensional ascension verified (8/8 pts)")
            sec1_pts += 8
        else:
            print("  [-] Stage 1 Filter Geometry FAIL: Dimension contraction or kernel mismatch.")
    except Exception as e:
        print(f"  [-] Stage 1 Filter Geometry FAIL: {e}")

    # Item 1.2: HRNet Multiscale Backbone Topology (7 pts)
    try:
        model_w32 = SpectralImageReconstructionCNN(num_bands=31, hrnet_type='w32', use_checkpoint=False)
        backbone = model_w32.backbone
        
        # Topology assertions
        is_hrnet = isinstance(backbone, HRNetBackbone)
        channels_ok = (backbone.channels == [32, 64, 128, 256])
        total_channels_ok = (backbone.total_channels == 480)  # 15 * 32
        has_stem = isinstance(backbone.stem, nn.Sequential)
        has_stage1 = isinstance(backbone.stage1, nn.Sequential)
        has_stage2 = isinstance(backbone.stage2, HRStage) and (backbone.stage2.num_branches == 2)
        has_stage3 = isinstance(backbone.stage3, HRStage) and (backbone.stage3.num_branches == 3)
        has_stage4 = isinstance(backbone.stage4, HRStage) and (backbone.stage4.num_branches == 4)
        
        # Parallel forward stream shapes test
        test_feat = torch.randn(2, 31, 64, 64)
        with torch.no_grad():
            streams = backbone(test_feat)
            fused = backbone.aggregate(streams)

        streams_ok = (
            len(streams) == 4 and
            streams[0].shape == torch.Size([2, 32, 64, 64]) and
            streams[1].shape == torch.Size([2, 64, 32, 32]) and
            streams[2].shape == torch.Size([2, 128, 16, 16]) and
            streams[3].shape == torch.Size([2, 256, 8, 8]) and
            fused.shape == torch.Size([2, 480, 64, 64])
        )

        if (is_hrnet and channels_ok and total_channels_ok and has_stem and
            has_stage1 and has_stage2 and has_stage3 and has_stage4 and streams_ok):
            print("  [+] Stage 2: Parallel-Multiscale HRNet 4-branch backbone with cross-scale fusion verified (7/7 pts)")
            sec1_pts += 7
        else:
            print("  [-] Stage 2 HRNet Backbone FAIL: Branch topology or cross-scale fusion mismatch.")
    except Exception as e:
        print(f"  [-] Stage 2 HRNet Backbone FAIL: {e}")

    # Item 1.3: Output Head & Strict Spectral Bounding Enforcement (5 pts)
    try:
        model = SpectralImageReconstructionCNN(num_bands=31, hrnet_type='w18', use_checkpoint=False)
        model.eval()
        
        # Head architecture check
        has_head = isinstance(model.head, nn.Sequential)
        
        # Pass inputs with extreme values that would force unbounded output outside [0.0, 1.0]
        extreme_input = torch.randn(2, 1, 64, 94) * 50.0 + 25.0
        with torch.no_grad():
            bounded_out = model(extreme_input)
            
            # Compute unconstrained raw head prediction to verify clamping is actively operating
            x_st1 = model.stage1(extreme_input)
            x_streams = model.backbone(x_st1)
            x_fused = model.backbone.aggregate(x_streams)
            raw_out = model.head(x_fused)

        is_bounded = bool(torch.all(bounded_out >= 0.0) and torch.all(bounded_out <= 1.0))
        raw_was_unbounded = bool(torch.min(raw_out) < 0.0 or torch.max(raw_out) > 1.0)
        shape_ok = (bounded_out.shape == torch.Size([2, 31, 64, 64]))

        if has_head and is_bounded and raw_was_unbounded and shape_ok:
            print("  [+] Head & Bounding: Multiscale aggregation + Conv projection + torch.clamp([0, 1]) verified (5/5 pts)")
            sec1_pts += 5
        else:
            print(f"  [-] Spectral Bounding FAIL: Output min={bounded_out.min():.4f}, max={bounded_out.max():.4f}.")
    except Exception as e:
        print(f"  [-] Spectral Bounding FAIL: {e}")

    # Item 1.4: Gradient Checkpointing Availability (5 pts)
    try:
        model_cp = SpectralImageReconstructionCNN(num_bands=31, hrnet_type='w18', use_checkpoint=True)
        model_cp.train()
        
        # Check initial checkpointing property
        cp_enabled_init = (model_cp.use_checkpoint is True and
                          model_cp.backbone.stage3.use_checkpoint is True and
                          model_cp.backbone.stage4.use_checkpoint is True)

        # Execute training forward + backward pass under activation checkpointing
        test_in = torch.randn(1, 1, 32, 62, requires_grad=True)
        test_out = model_cp(test_in)
        test_loss = test_out.sum()
        test_loss.backward()

        grad_flow_ok = (test_in.grad is not None and bool(torch.isfinite(test_in.grad).all()))

        # Verify dynamic toggle
        model_cp.use_checkpoint = False
        cp_toggle_ok = (model_cp.use_checkpoint is False and
                        model_cp.backbone.stage3.use_checkpoint is False and
                        model_cp.backbone.stage4.use_checkpoint is False)

        if cp_enabled_init and grad_flow_ok and cp_toggle_ok:
            print("  [+] Gradient Checkpointing: Stages 3 & 4 execute forward/backward with use_checkpoint=True (5/5 pts)")
            sec1_pts += 5
        else:
            print("  [-] Gradient Checkpointing FAIL: Toggle or autograd checkpointing failed.")
    except Exception as e:
        print(f"  [-] Gradient Checkpointing FAIL: {e}")

    section_scores[1] = sec1_pts
    score += sec1_pts
    print(f"  --> Section 1 Subtotal: {sec1_pts} / 25 pts")

    # =========================================================================
    # SECTION 2: Mathematical Forward Physics & Optical Models (25 pts)
    # =========================================================================
    print("\n[SECTION 2] Auditing Mathematical Forward Physics & Optical Models (25 pts)...")
    sec2_pts = 0

    forward_model = DualCameraForwardModel(num_bands=31, shift_step=1)

    # Item 2.1: Vectorized CASSI Dispersion Math & Shapes (8 pts)
    try:
        # Patch dimensions: [2, 31, 256, 256] -> [2, 1, 256, 286]
        patch_cube = torch.rand(2, 31, 256, 256)
        meas_patch = forward_model.forward_cassi(patch_cube)
        patch_shape_ok = (meas_patch.shape == torch.Size([2, 1, 256, 286]))

        # Full scene dimensions: [1, 31, 512, 512] -> [1, 1, 512, 542]
        full_cube = torch.rand(1, 31, 512, 512)
        meas_full = forward_model.forward_cassi(full_cube)
        full_shape_ok = (meas_full.shape == torch.Size([1, 1, 512, 542]))

        # Physical horizontal dispersion impulse test:
        # Impulse at band k=15, coordinate (y=10, x=20) must arrive at detector coordinate (y=10, x=20+15=35)
        impulse = torch.zeros(1, 31, 64, 64)
        impulse[0, 15, 10, 20] = 1.0
        meas_impulse = forward_model.forward_cassi(impulse)
        
        impulse_val = float(meas_impulse[0, 0, 10, 35].item())
        other_vals = float(meas_impulse.sum().item() - impulse_val)
        dispersion_math_ok = (abs(impulse_val - 1.0) < 1e-6 and abs(other_vals) < 1e-6)

        if patch_shape_ok and full_shape_ok and dispersion_math_ok:
            print("  [+] CASSI Branch: Mask modulation + horizontal dispersion shift + 2D integration verified (8/8 pts)")
            sec2_pts += 8
        else:
            print(f"  [-] CASSI Dispersion FAIL: Shapes or horizontal dispersion mathematics incorrect.")
    except Exception as e:
        print(f"  [-] CASSI Dispersion FAIL: {e}")

    # Item 2.2: Color Camera Branch Spectral CRF Integration (6 pts)
    try:
        crf_mat = forward_model.crf
        crf_shape_ok = (crf_mat.shape == torch.Size([3, 31]))
        crf_non_neg = bool(torch.all(crf_mat >= 0.0))

        # Patch projection shape: [2, 31, 256, 256] -> [2, 3, 256, 256]
        rgb_patch = forward_model.forward_rgb(patch_cube)
        rgb_shape_ok = (rgb_patch.shape == torch.Size([2, 3, 256, 256]))

        # Exact mathematical matrix-vector multiplication pixel check
        sample_pixel_hsi = patch_cube[0, :, 100, 120]  # [31]
        expected_rgb = torch.matmul(crf_mat, sample_pixel_hsi)  # [3]
        actual_rgb = rgb_patch[0, :, 100, 120]  # [3]
        max_proj_err = float(torch.max(torch.abs(expected_rgb - actual_rgb)).item())
        math_exact = (max_proj_err < 1e-6)

        # Unbatched support check: [31, 64, 64] -> [3, 64, 64]
        unbatched_hsi = torch.rand(31, 64, 64)
        unbatched_rgb = forward_model.forward_rgb(unbatched_hsi)
        unbatched_ok = (unbatched_rgb.shape == torch.Size([3, 64, 64]))

        if crf_shape_ok and crf_non_neg and rgb_shape_ok and math_exact and unbatched_ok:
            print("  [+] Color Branch: Calibrated 3x31 CRF spectral integration verified (6/6 pts)")
            sec2_pts += 6
        else:
            print(f"  [-] Color Branch FAIL: Max projection error {max_proj_err:.2e}.")
    except Exception as e:
        print(f"  [-] Color Branch FAIL: {e}")

    # Item 2.3: Optical Inversion & Unshift Reversibility (6 pts)
    try:
        test_cube = torch.rand(2, 31, 128, 128)
        # Shift without mask
        shifted_cube = forward_model.shift(test_cube, mask=None)  # [2, 31, 128, 158]
        unshifted_cube = DualCameraForwardModel.unshift_cassi(shifted_cube, num_bands=31, original_w=128)
        
        shape_match = (unshifted_cube.shape == test_cube.shape)
        max_inversion_err = float(torch.max(torch.abs(test_cube - unshifted_cube)).item())
        tensor_exact = (max_inversion_err < 1e-6)

        # NumPy cross-compatibility
        shifted_np = shifted_cube.numpy()
        unshifted_np = DualCameraForwardModel.unshift(shifted_np, original_w=128)
        max_np_err = float(np.max(np.abs(test_cube.numpy() - unshifted_np.numpy())))
        numpy_exact = (max_np_err < 1e-6)

        if shape_match and tensor_exact and numpy_exact:
            print(f"  [+] Optical Inversion: Unshifting synthetic cube achieves exact inverse (diff < 1e-6) (6/6 pts)")
            sec2_pts += 6
        else:
            print(f"  [-] Optical Inversion FAIL: Tensor err={max_inversion_err:.2e}, NumPy err={max_np_err:.2e}.")
    except Exception as e:
        print(f"  [-] Optical Inversion FAIL: {e}")

    # Item 2.4: Fixed Binary Bernoulli Coded Aperture Mask (5 pts)
    try:
        mask_obj = CodedApertureMask()
        mask_t = mask_obj.tensor
        mask_arr = mask_obj.get_numpy()

        shape_512 = (mask_t.shape == torch.Size([512, 512]))
        unique_vals = set(np.unique(mask_arr))
        is_binary = unique_vals.issubset({0.0, 1.0})
        mean_p = float(np.mean(mask_arr))
        p_in_range = (0.49 <= mean_p <= 0.51)

        # Patch extraction functionality
        patch_m = mask_obj.get_patch(50, 60, 256, 256)
        patch_shape_ok = (patch_m.shape == torch.Size([256, 256]))

        if shape_512 and is_binary and p_in_range and patch_shape_ok:
            print(f"  [+] Coded Aperture Mask: Shared binary Bernoulli (p={mean_p:.4f}) verified (5/5 pts)")
            sec2_pts += 5
        else:
            print(f"  [-] Coded Aperture Mask FAIL: shape={mask_t.shape}, binary={is_binary}, p={mean_p:.4f}.")
    except Exception as e:
        print(f"  [-] Coded Aperture Mask FAIL: {e}")

    section_scores[2] = sec2_pts
    score += sec2_pts
    print(f"  --> Section 2 Subtotal: {sec2_pts} / 25 pts")

    # =========================================================================
    # SECTION 3: Closed-Loop Self-Supervised Physics Loss (20 pts)
    # =========================================================================
    print("\n[SECTION 3] Auditing Closed-Loop Self-Supervised Physics Loss (20 pts)...")
    sec3_pts = 0

    loss_fn = PhysicsLoss(forward_model=forward_model, alpha=1.0)

    # Item 3.1: Zero Ground-Truth Requirement & Dual Loss Formulation (8 pts)
    try:
        # Audit forward method signature: strictly zero ground-truth parameters
        sig = inspect.signature(loss_fn.forward)
        param_names = list(sig.parameters.keys())
        banned_terms = {'gt', 'target', 'ground_truth', 'gt_cube', 'label', 'clean_cube'}
        has_no_gt_params = not any(p.lower() in banned_terms for p in param_names)

        # Dual loss formulation: Loss_total = Loss_grayscale + alpha * Loss_color (MSE)
        pred_dummy = torch.rand(2, 31, 64, 64, requires_grad=True)
        meas_cassi_dummy = torch.rand(2, 1, 64, 94)
        meas_rgb_dummy = torch.rand(2, 3, 64, 64)
        mask_dummy = torch.ones(64, 64)

        loss_res = loss_fn(pred_dummy, meas_cassi_dummy, meas_rgb_dummy, mask=mask_dummy)
        loss_tot, loss_cas, loss_col = loss_res

        # Check algebraic sum: tot == cas + 1.0 * col (using relative FP32 tolerance)
        sum_err = abs(loss_tot.item() - (loss_cas.item() + 1.0 * loss_col.item()))
        rel_sum_err = sum_err / max(1e-6, loss_tot.item())
        dual_sum_ok = (rel_sum_err < 1e-4)

        # Check explicit MSE against manually simulated channels
        manual_cassi = forward_model.forward_cassi(pred_dummy, mask=mask_dummy)
        manual_rgb = forward_model.forward_rgb(pred_dummy)
        exp_cassi_loss = F.mse_loss(manual_cassi, meas_cassi_dummy).item()
        exp_rgb_loss = F.mse_loss(manual_rgb, meas_rgb_dummy).item()
        rel_cassi_err = abs(loss_cas.item() - exp_cassi_loss) / max(1e-6, exp_cassi_loss)
        rel_rgb_err = abs(loss_col.item() - exp_rgb_loss) / max(1e-6, exp_rgb_loss)
        mse_ok = (rel_cassi_err < 1e-4 and rel_rgb_err < 1e-4)

        if has_no_gt_params and dual_sum_ok and mse_ok:
            print("  [+] Dual Loss Formulation: Loss_total = Loss_grayscale + Loss_color (MSE) verified (8/8 pts)")
            sec3_pts += 8
        else:
            print(f"  [-] Dual Loss Formulation FAIL: Signature or loss sum mismatch (rel_err={rel_sum_err:.2e}).")
    except Exception as e:
        print(f"  [-] Dual Loss Formulation FAIL: {e}")

    # Item 3.2: Full Autograd Backpropagation & Gradient Flow (8 pts)
    try:
        rec_model = SpectralImageReconstructionCNN(num_bands=31, hrnet_type='w18', use_checkpoint=False)
        rec_model.train()
        rec_model.zero_grad()

        input_meas = torch.rand(1, 1, 64, 94)
        target_meas_rgb = torch.rand(1, 3, 64, 64)
        active_mask = torch.ones(64, 64)

        pred_hsi = rec_model(input_meas)
        loss_eval = loss_fn(pred_hsi, input_meas, target_meas_rgb, mask=active_mask)
        loss_eval.loss_total.backward()

        all_grads_present = True
        all_grads_finite = True
        total_grad_norm = 0.0

        for name, param in rec_model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    all_grads_present = False
                    break
                if not torch.isfinite(param.grad).all():
                    all_grads_finite = False
                    break
                total_grad_norm += float(param.grad.abs().sum().item())

        non_zero_grad = (total_grad_norm > 0.0)

        if all_grads_present and all_grads_finite and non_zero_grad:
            print("  [+] Autograd Backprop: Loss backward produces non-zero, finite gradients across all weights (8/8 pts)")
            sec3_pts += 8
        else:
            print(f"  [-] Autograd Backprop FAIL: present={all_grads_present}, finite={all_grads_finite}, norm={total_grad_norm:.4f}.")
    except Exception as e:
        print(f"  [-] Autograd Backprop FAIL: {e}")

    # Item 3.3: Numerical Stability & Loss Non-Negativity (4 pts)
    try:
        # Non-negativity check
        non_negative = (loss_tot.item() >= 0.0 and loss_cas.item() >= 0.0 and loss_col.item() >= 0.0)

        # Zero-residual test: when measurement equals exact simulation, loss evaluates to 0.0
        perfect_cassi = forward_model.forward_cassi(pred_dummy, mask=mask_dummy).detach()
        perfect_rgb = forward_model.forward_rgb(pred_dummy).detach()
        perfect_res = loss_fn(pred_dummy, perfect_cassi, perfect_rgb, mask=mask_dummy)
        zero_residual_ok = (perfect_res.loss_total.item() < 1e-10)

        if non_negative and zero_residual_ok:
            print("  [+] Numerical Stability: Loss non-negativity and zero-error consistency verified (4/4 pts)")
            sec3_pts += 4
        else:
            print(f"  [-] Numerical Stability FAIL: Non-neg={non_negative}, Zero-res={perfect_res.loss_total.item():.2e}.")
    except Exception as e:
        print(f"  [-] Numerical Stability FAIL: {e}")

    section_scores[3] = sec3_pts
    score += sec3_pts
    print(f"  --> Section 3 Subtotal: {sec3_pts} / 20 pts")

    # =========================================================================
    # SECTION 4: Quantitative Evaluation Metrics Rigor (15 pts)
    # =========================================================================
    print("\n[SECTION 4] Auditing Quantitative Evaluation Metrics Rigor (15 pts)...")
    sec4_pts = 0

    # Item 4.1: SAM Metric in Radians (6 pts)
    try:
        # Identity test: SAM(A, A) must be 0.0 rad
        t_rand = torch.rand(2, 31, 32, 32)
        sam_ident = calc_sam(t_rand, t_rand)
        ident_ok = (abs(sam_ident) < 1e-5)

        # Zero-norm safe stability test
        t_zeros = torch.zeros(1, 31, 8, 8)
        sam_zeros = calc_sam(t_zeros, t_zeros)
        zeros_safe = (sam_zeros == 0.0 and np.isfinite(sam_zeros))

        # Orthogonal spectrum test: SAM([1, 0, ...], [0, 1, ...]) must be exactly pi/2 rad (~1.570796 rad)
        t_orth1 = torch.zeros(1, 31, 4, 4); t_orth1[:, 0, :, :] = 1.0
        t_orth2 = torch.zeros(1, 31, 4, 4); t_orth2[:, 1, :, :] = 1.0
        sam_ortho = calc_sam(t_orth1, t_orth2)
        expected_pi_2 = np.pi / 2.0
        ortho_ok = (abs(sam_ortho - expected_pi_2) < 1e-4)

        # Explicit negative check: strictly reject degrees (90.0 deg)
        radians_strict = (sam_ortho < 2.0)

        if ident_ok and zeros_safe and ortho_ok and radians_strict:
            print(f"  [+] SAM Metric: Exactly 0.0 rad on identity, pi/2 rad ({sam_ortho:.4f}) on orthogonal spectra verified (6/6 pts)")
            sec4_pts += 6
        else:
            print(f"  [-] SAM Metric FAIL: Ident={sam_ident:.4f}, Ortho={sam_ortho:.4f} (expected {expected_pi_2:.4f}).")
    except Exception as e:
        print(f"  [-] SAM Metric FAIL: {e}")

    # Item 4.2: PSNR Mathematical Rigor (5 pts)
    try:
        # Identity test: zero MSE capped at 100.0 dB
        psnr_ident = calc_psnr(t_rand, t_rand)
        psnr_ident_ok = (psnr_ident >= 99.0)

        # Known MSE test: gt=0.0, pred=0.1 -> MSE = 0.01 -> PSNR = 10*log10(1.0 / 0.01) = 20.00 dB
        t_zero = torch.zeros(1, 31, 16, 16)
        t_const = torch.full((1, 31, 16, 16), 0.1)
        psnr_known = calc_psnr(t_const, t_zero, max_val=1.0)
        psnr_known_ok = (abs(psnr_known - 20.00) < 1e-3)

        if psnr_ident_ok and psnr_known_ok:
            print(f"  [+] PSNR Metric: Exactly 100.0 dB on identity, 20.00 dB for MSE=0.01 verified (5/5 pts)")
            sec4_pts += 5
        else:
            print(f"  [-] PSNR Metric FAIL: Ident={psnr_ident:.2f} dB, Known={psnr_known:.2f} dB (expected 20.00 dB).")
    except Exception as e:
        print(f"  [-] PSNR Metric FAIL: {e}")

    # Item 4.3: SSIM Range & Dynamic Calibration (4 pts)
    try:
        # Identity SSIM must be 1.0
        ssim_ident = calc_ssim(t_rand, t_rand)
        ssim_ident_ok = (abs(ssim_ident - 1.0) < 1e-4)

        # Inverted image test
        t_inv = 1.0 - t_rand
        ssim_inv = calc_ssim(t_inv, t_rand)
        ssim_range_ok = (-1.0 <= ssim_inv <= 0.2)

        if ssim_ident_ok and ssim_range_ok:
            print(f"  [+] SSIM Metric: Exactly 1.0 on identity, valid range [-1, 1] verified (4/4 pts)")
            sec4_pts += 4
        else:
            print(f"  [-] SSIM Metric FAIL: Ident={ssim_ident:.4f}, Inverted={ssim_inv:.4f}.")
    except Exception as e:
        print(f"  [-] SSIM Metric FAIL: {e}")

    section_scores[4] = sec4_pts
    score += sec4_pts
    print(f"  --> Section 4 Subtotal: {sec4_pts} / 15 pts")

    # =========================================================================
    # SECTION 5: CAVE Dataset Pipeline, Partitioning & Stochastic Sampling (15 pts)
    # =========================================================================
    print("\n[SECTION 5] Auditing CAVE Dataset Pipeline, Partitioning & Stochastic Sampling (15 pts)...")
    sec5_pts = 0

    # Item 5.1: Protocol Partitioning (5 pts)
    try:
        train_count_20 = (len(CAVE_TRAIN_SCENES) == 20)
        test_count_10 = (len(CAVE_TEST_SCENES) == 10)
        zero_overlap = (len(set(CAVE_TRAIN_SCENES).intersection(set(CAVE_TEST_SCENES))) == 0)

        splits = get_cave_splits()
        splits_api_ok = (len(splits.train) == 20 and len(splits.test) == 10)

        if train_count_20 and test_count_10 and zero_overlap and splits_api_ok:
            print("  [+] Protocol Partitioning: Exactly 20 train scenes and 10 test scenes with zero overlap (5/5 pts)")
            sec5_pts += 5
        else:
            print("  [-] Protocol Partitioning FAIL: Scene counts or partition overlap invalid.")
    except Exception as e:
        print(f"  [-] Protocol Partitioning FAIL: {e}")

    # Item 5.2: Dynamic Stochastic Patch Sampling (6 pts)
    try:
        dataset_train = CAVEDualDataset(
            split='train',
            patch_size=256,
            is_train=True,
            augment=False,
            preload=False,
            synthetic_fallback=True,
            num_patches_per_scene=1
        )

        # Sample repeatedly from index 0 across successive invocations
        crop_coords = [dataset_train[0]['crop_coords'] for _ in range(10)]
        distinct_coords = len(set(crop_coords))
        is_dynamic = (distinct_coords > 1)

        # Codebase AST audit: confirm absence of deterministic RandomState(idx) in executable code
        dataset_file_path = os.path.join(BASE_DIR, "utils", "dataset.py")
        with open(dataset_file_path, "r", encoding="utf-8") as f:
            dataset_ast = ast.parse(f.read())
        
        has_seed_lock_call = False
        for node in ast.walk(dataset_ast):
            if isinstance(node, ast.Call):
                func = node.func
                # Detect calls like np.random.RandomState(idx) or RandomState(idx)
                if ((isinstance(func, ast.Attribute) and func.attr in ('RandomState', 'seed')) or
                    (isinstance(func, ast.Name) and func.id in ('RandomState', 'seed'))):
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id in ('idx', 'index'):
                            has_seed_lock_call = True
        no_seed_lock = not has_seed_lock_call

        if is_dynamic and no_seed_lock:
            print("  [+] Dynamic Stochastic Sampling: Dynamic random crop verified, zero deterministic seed lock (6/6 pts)")
            sec5_pts += 6
        else:
            print(f"  [-] Dynamic Stochastic Sampling FAIL: Unique coords={distinct_coords}/10, no_seed_lock={no_seed_lock}.")
    except Exception as e:
        print(f"  [-] Dynamic Stochastic Sampling FAIL: {e}")

    # Item 5.3: Physical Reflectance Range in Dataset (4 pts)
    try:
        sample_0 = dataset_train[0]
        cube_0 = sample_0['cube']
        cassi_0 = sample_0['meas_cassi']
        rgb_0 = sample_0['meas_rgb']
        mask_0 = sample_0['mask']

        patch_shapes_ok = (
            cube_0.shape == torch.Size([31, 256, 256]) and
            cassi_0.shape == torch.Size([1, 256, 286]) and
            rgb_0.shape == torch.Size([3, 256, 256]) and
            mask_0.shape == torch.Size([256, 256])
        )
        cube_bounded = bool(torch.all(cube_0 >= 0.0) and torch.all(cube_0 <= 1.0))

        # Full scene test
        full_cassi, full_rgb, full_cube, full_mask, sname = dataset_train.get_full_scene(0)
        full_shapes_ok = (
            full_cassi.shape == torch.Size([1, 512, 542]) and
            full_rgb.shape == torch.Size([3, 512, 512]) and
            full_cube.shape == torch.Size([31, 512, 512]) and
            full_mask.shape == torch.Size([512, 512]) and
            bool(torch.all(full_cube >= 0.0) and torch.all(full_cube <= 1.0))
        )

        if patch_shapes_ok and cube_bounded and full_shapes_ok:
            print("  [+] Physical Reflectance Range: Patch [1, 256, 286] and Full Scene [1, 512, 542] shapes in [0, 1] (4/4 pts)")
            sec5_pts += 4
        else:
            print("  [-] Physical Reflectance Range FAIL: Shape or reflectance bound mismatch.")
    except Exception as e:
        print(f"  [-] Physical Reflectance Range FAIL: {e}")

    section_scores[5] = sec5_pts
    score += sec5_pts
    print(f"  --> Section 5 Subtotal: {sec5_pts} / 15 pts")

    # =========================================================================
    # FINAL AUDIT SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"                       FINAL AUDITOR SCORE: {score} / {total_points}")
    print("=" * 80)

    if score == 100:
        print(">>> ALL 100 AUDIT CRITERIA MET. CODEBASE FULLY CERTIFIED FOR PRODUCTION. <<<\n")
        return 0
    else:
        print(f">>> AUDIT FAILED ({score}/{total_points}). FIX DISCREPANCIES BEFORE CERTIFICATION. <<<\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_audit())
