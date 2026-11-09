"""
Adversarial Stress Testing Harness for Xie_Rep_2
Challenger 1: Optical Physics, Dispersion Geometry, and Stage 1 Filter Stress-Testing
"""

import sys
import os
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root is in path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.sir_cnn import SpectralImageReconstructionCNN, Stage1DimensionalAscending
from models.forward_model import DualCameraForwardModel
from models.crf import CameraResponseFunction, get_crf_matrix, project_rgb, REGISTERED_CRF_PROFILES
from models.mask import CodedApertureMask, modulate_cube, generate_bernoulli_mask
from losses.physics_loss import PhysicsLoss, LossResult


class TestAdversarialStage1(unittest.TestCase):
    """Stress tests on Stage 1 (1xλ) convolution layer."""

    def setUp(self):
        self.num_bands = 31
        self.stage1 = Stage1DimensionalAscending(in_channels=1, num_bands=self.num_bands)

    def test_filter_geometry(self):
        conv = self.stage1.conv1x_lambda
        self.assertIsInstance(conv, nn.Conv2d)
        self.assertEqual(conv.in_channels, 1)
        self.assertEqual(conv.out_channels, 31)
        self.assertEqual(conv.kernel_size, (1, 31))
        self.assertEqual(conv.stride, (1, 1))
        self.assertEqual(conv.padding, (0, 0))
        self.assertNotEqual(conv.kernel_size, (1, 1), "Pointwise 1x1 conv detected!")

    def test_extreme_shapes(self):
        shapes_and_expected = [
            ((1, 1, 64, 94), (1, 31, 64, 64)),
            ((4, 1, 128, 158), (4, 31, 128, 128)),
            ((2, 1, 256, 286), (2, 31, 256, 256)),
            ((1, 1, 512, 542), (1, 31, 512, 512)),
            # Non-square / arbitrary dimensions
            ((1, 1, 17, 75), (1, 31, 17, 45)),
            ((3, 1, 33, 40), (3, 31, 33, 10)),
            # Minimal width: W_disp = 31 -> W_out = 1
            ((1, 1, 16, 31), (1, 31, 16, 1)),
        ]
        for in_shape, exp_shape in shapes_and_expected:
            x = torch.randn(*in_shape)
            out = self.stage1(x)
            self.assertEqual(out.shape, torch.Size(exp_shape), f"Shape mismatch for input {in_shape}")

    def test_under_width_fails(self):
        # Input with width < 31 must raise an error in Conv2d
        x_too_narrow = torch.randn(1, 1, 16, 30)
        with self.assertRaises(RuntimeError):
            self.stage1(x_too_narrow)

    def test_non_standard_values(self):
        # 1. Very large numbers (1e6)
        x_large = torch.full((1, 1, 32, 62), 1e6)
        out_large = self.stage1(x_large)
        self.assertTrue(torch.isfinite(out_large).all())
        self.assertTrue((out_large >= 0.0).all(), "ReLU violation on large numbers")

        # 2. Near-zero numbers (1e-15)
        x_tiny = torch.full((1, 1, 32, 62), 1e-15)
        out_tiny = self.stage1(x_tiny)
        self.assertTrue(torch.isfinite(out_tiny).all())
        self.assertTrue((out_tiny >= 0.0).all(), "ReLU violation on tiny numbers")

        # 3. Negative values (-1000.0) -> must produce non-negative outputs due to ReLU
        x_neg = torch.full((1, 1, 32, 62), -1000.0)
        out_neg = self.stage1(x_neg)
        self.assertTrue(torch.isfinite(out_neg).all())
        self.assertTrue((out_neg >= 0.0).all(), "ReLU must ensure all outputs >= 0.0")

        # 4. Mixed extreme values
        x_mixed = torch.tensor([-1e6, -100.0, 0.0, 1e-10, 100.0, 1e6]).repeat(1, 1, 10, 11)[:, :, :32, :62]
        out_mixed = self.stage1(x_mixed)
        self.assertTrue(torch.isfinite(out_mixed).all())
        self.assertTrue((out_mixed >= 0.0).all(), "ReLU must ensure all outputs >= 0.0")


class TestAdversarialForwardPhysics(unittest.TestCase):
    """Stress tests on CASSI forward physics, Kronecker delta spikes, unshifting, and CRF projection."""

    def setUp(self):
        self.fwd = DualCameraForwardModel(num_bands=31, shift_step=1)

    def test_kronecker_delta_spikes(self):
        """
        Kronecker delta spikes at arbitrary (λ, y, x) must land on detector at (y, x + λ - 1)
        where λ is 1-indexed, i.e., channel index k in [0, 30] lands at x + k.
        """
        H, W = 64, 64
        rng = np.random.RandomState(1337)
        for trial in range(10):
            k = int(rng.randint(0, 31))
            y = int(rng.randint(0, H))
            x = int(rng.randint(0, W))

            cube = torch.zeros(1, 31, H, W)
            cube[0, k, y, x] = 1.0

            meas = self.fwd.forward_cassi(cube, mask=None)
            exp_col = x + k
            self.assertEqual(meas.shape, torch.Size([1, 1, H, W + 30]))
            
            val = float(meas[0, 0, y, exp_col].item())
            self.assertAlmostEqual(val, 1.0, places=6, msg=f"Impulse at band {k}, (y={y}, x={x}) did not reach 1.0 at col {exp_col}")
            
            total_sum = float(meas.sum().item())
            self.assertAlmostEqual(total_sum, 1.0, places=6, msg=f"Energy leaked: total sum {total_sum} != 1.0")

    def test_kronecker_delta_with_mask(self):
        """
        With a binary mask, impulse is multiplied by mask[y, x].
        """
        H, W = 64, 64
        mask = torch.zeros(H, W)
        mask[10, 20] = 1.0  # open aperture
        mask[10, 25] = 0.0  # closed aperture

        # Case A: open aperture
        cube_open = torch.zeros(1, 31, H, W)
        cube_open[0, 5, 10, 20] = 1.0
        meas_open = self.fwd.forward_cassi(cube_open, mask=mask)
        self.assertAlmostEqual(meas_open[0, 0, 10, 20 + 5].item(), 1.0, places=6)

        # Case B: closed aperture
        cube_closed = torch.zeros(1, 31, H, W)
        cube_closed[0, 5, 10, 25] = 1.0
        meas_closed = self.fwd.forward_cassi(cube_closed, mask=mask)
        self.assertAlmostEqual(meas_closed.sum().item(), 0.0, places=6)

    def test_unshifting_reversibility(self):
        """
        Test that unshift(shift(cube)) == cube with error < 1e-6.
        """
        configs = [
            (2, 31, 64, 64),
            (1, 31, 128, 128),
            (2, 31, 256, 256),
            (1, 31, 512, 512),
        ]
        for b, c, h, w in configs:
            orig_cube = torch.rand(b, c, h, w)
            shifted = self.fwd.shift(orig_cube, mask=None)
            self.assertEqual(shifted.shape, torch.Size([b, c, h, w + 30]))

            unshifted = DualCameraForwardModel.unshift_cassi(shifted, num_bands=31, original_w=w)
            self.assertEqual(unshifted.shape, orig_cube.shape)

            max_diff = (orig_cube - unshifted).abs().max().item()
            self.assertLess(max_diff, 1e-6, f"Reversibility error {max_diff:.2e} >= 1e-6 for shape ({b},{c},{h},{w})")

    def test_unshifting_numpy(self):
        """Test unshifting on numpy arrays."""
        orig_np = np.random.rand(1, 31, 64, 64).astype(np.float32)
        shifted_t = self.fwd.shift(torch.from_numpy(orig_np), mask=None)
        shifted_np = shifted_t.numpy()

        unshifted_np = DualCameraForwardModel.unshift(shifted_np, original_w=64)
        max_diff = np.max(np.abs(orig_np - unshifted_np.numpy()))
        self.assertLess(max_diff, 1e-6, f"NumPy unshifting error {max_diff:.2e} >= 1e-6")

    def test_unshifting_1channel_meas(self):
        """
        Test unshifting directly on 1-channel compressive snapshot measurement [B, 1, H, total_W].
        When only band 0 is active, band 0 of unshifted measurement must recover original band 0.
        """
        H, W = 64, 64
        cube = torch.zeros(1, 31, H, W)
        cube[0, 0, :, :] = torch.rand(H, W)  # only band 0 is active

        meas = self.fwd.forward_cassi(cube, mask=None)  # [1, 1, H, W + 30]
        unshifted = DualCameraForwardModel.unshift_cassi(meas, num_bands=31, original_w=W)
        self.assertEqual(unshifted.shape, torch.Size([1, 31, H, W]))

        diff_band0 = (unshifted[0, 0] - cube[0, 0]).abs().max().item()
        self.assertLess(diff_band0, 1e-6)

    def test_unshift_invalid_dims_raise(self):
        """Assert unshift raises ValueError on contradictory dimensions."""
        invalid_tensor = torch.zeros(1, 1, 64, 20)  # total_W = 20 < 31
        with self.assertRaises(ValueError):
            DualCameraForwardModel.unshift_cassi(invalid_tensor, num_bands=31, original_w=64)

    def test_crf_dot_product_equivalence(self):
        """
        Assert mathematical equivalence of forward_rgb to manual pixel-wise dot products across all registered CRF profiles.
        """
        for profile_name in REGISTERED_CRF_PROFILES.keys():
            crf_mat = get_crf_matrix(profile_name)
            fwd_custom = DualCameraForwardModel(crf_matrix=crf_mat)

            cube = torch.rand(2, 31, 32, 32)
            rgb_fwd = fwd_custom.forward_rgb(cube)
            self.assertEqual(rgb_fwd.shape, torch.Size([2, 3, 32, 32]))

            expected_rgb = torch.einsum('cl, blhw -> bchw', crf_mat, cube)
            diff = (rgb_fwd - expected_rgb).abs().max().item()
            self.assertLess(diff, 1e-6, f"CRF projection discrepancy for {profile_name}: {diff:.2e}")

            sample_pix = cube[0, :, 15, 20]
            for c in range(3):
                expected_c = torch.dot(crf_mat[c], sample_pix).item()
                actual_c = rgb_fwd[0, c, 15, 20].item()
                self.assertAlmostEqual(actual_c, expected_c, places=6)


class TestAdversarialSpectralBounding(unittest.TestCase):
    """Stress tests on Model Spectral Bounding under extreme inputs."""

    def test_extreme_inputs_bounded(self):
        """
        Feed synthetic inputs with magnitude +1000.0, -1000.0, and mixed extremes
        to SpectralImageReconstructionCNN and assert outputs are strictly within [0.0, 1.0].
        """
        model = SpectralImageReconstructionCNN(num_bands=31, hrnet_type='w18', use_checkpoint=False)
        model.eval()

        extreme_test_cases = [
            ("all_plus_1000", torch.full((1, 1, 64, 94), 1000.0)),
            ("all_minus_1000", torch.full((1, 1, 64, 94), -1000.0)),
            ("large_positive_noise", torch.rand(1, 1, 64, 94) * 1000.0),
            ("large_negative_noise", -torch.rand(1, 1, 64, 94) * 1000.0),
            ("extreme_bipolar", (torch.randint(0, 2, (2, 1, 64, 94)).float() * 2 - 1) * 1000.0),
            ("huge_magnitudes", torch.randn(1, 1, 64, 94) * 1e5),
        ]

        for case_name, x_in in extreme_test_cases:
            with torch.no_grad():
                out = model(x_in)
                min_val = out.min().item()
                max_val = out.max().item()

                self.assertTrue(
                    min_val >= 0.0,
                    f"[{case_name}] Min value {min_val} < 0.0 violates spectral non-negativity"
                )
                self.assertTrue(
                    max_val <= 1.0,
                    f"[{case_name}] Max value {max_val} > 1.0 violates spectral upper bound"
                )
                self.assertTrue(
                    torch.isfinite(out).all(),
                    f"[{case_name}] Output contains NaN or Inf"
                )

    def test_clamping_is_actively_operating(self):
        """
        Verify that without the clamp, the raw unconstrained projection from the head
        exceeds [0.0, 1.0] under extreme input, confirming that torch.clamp is actively functioning.
        """
        model = SpectralImageReconstructionCNN(num_bands=31, hrnet_type='w18', use_checkpoint=False)
        model.eval()

        x_in = torch.randn(1, 1, 64, 94) * 100.0
        with torch.no_grad():
            x_st1 = model.stage1(x_in)
            streams = model.backbone(x_st1)
            fused = model.backbone.aggregate(streams)
            raw_head_out = model.head(fused)

            clamped_out = model(x_in)

            raw_exceeds = (raw_head_out.min().item() < 0.0 or raw_head_out.max().item() > 1.0)
            self.assertTrue(raw_exceeds, "Raw head output was already in [0, 1] unexpectedly")

            self.assertTrue((clamped_out >= 0.0).all().item())
            self.assertTrue((clamped_out <= 1.0).all().item())

    def test_w32_full_resolution_and_amp(self):
        """
        Test HRNet-W32 architecture with full patch [1, 1, 256, 286] -> [1, 31, 256, 256]
        under both standard and AMP contexts.
        """
        model = SpectralImageReconstructionCNN(num_bands=31, hrnet_type='w32', use_checkpoint=True)
        model.eval()

        x = torch.rand(1, 1, 256, 286)
        with torch.no_grad():
            out = model(x)
            self.assertEqual(out.shape, torch.Size([1, 31, 256, 256]))
            self.assertTrue((out >= 0.0).all().item())
            self.assertTrue((out <= 1.0).all().item())


class TestAdversarialMaskModule(unittest.TestCase):
    """Stress tests on CodedApertureMask and spatial modulation."""

    def test_mask_boundary_checks(self):
        mask_obj = CodedApertureMask(shape=(512, 512), p=0.5, seed=42)
        
        # Out of bounds vertical crop
        with self.assertRaises(ValueError):
            mask_obj.get_patch(h_start=300, w_start=0, patch_h=256, patch_w=256)
            
        # Out of bounds horizontal crop
        with self.assertRaises(ValueError):
            mask_obj.get_patch(h_start=0, w_start=300, patch_h=256, patch_w=256)

        # Negative coordinate
        with self.assertRaises(ValueError):
            mask_obj.get_patch(h_start=-5, w_start=0, patch_h=256, patch_w=256)

    def test_mask_seed_reproducibility(self):
        m1 = generate_bernoulli_mask(shape=(256, 256), seed=123)
        m2 = generate_bernoulli_mask(shape=(256, 256), seed=123)
        m3 = generate_bernoulli_mask(shape=(256, 256), seed=456)

        self.assertTrue(np.array_equal(m1, m2))
        self.assertFalse(np.array_equal(m1, m3))


if __name__ == "__main__":
    unittest.main()
