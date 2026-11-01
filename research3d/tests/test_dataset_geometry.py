"""Dataset-unit checks against real publisher camera metadata, not benchmark scores."""

import csv, os, unittest
from pathlib import Path
import numpy as np
from knownmap.dataset_geometry import (
    hypersim_intrinsics,
    radial_distance_to_z,
    gl_asset_pose_to_cv_meters,
    reduce_semantic_labels,
    hypersim_camera_model,
    hypersim_quantization_bound,
    hypersim_camera_orientation,
    hypersim_world_camera_model,
)


class LabelTests(unittest.TestCase):
    def test_ade_unknown_is_not_class_zero_and_last_class_is_retained(self):
        np.testing.assert_array_equal(
            reduce_semantic_labels(np.array([0, 1, 150, 255], np.uint8), "ade20k"),
            [255, 0, 149, 255],
        )

    def test_hypersim_signed_unknown_and_unknown_codes_fail_closed(self):
        np.testing.assert_array_equal(
            reduce_semantic_labels(np.array([-1, 1, 40], np.int16), "hypersim"),
            [255, 0, 39],
        )
        with self.assertRaises(AssertionError):
            reduce_semantic_labels(np.array([0], np.int16), "hypersim")

    def test_scaled_camera_axes_preserve_rays_and_shear_is_rejected(self):
        R, scale = hypersim_camera_orientation(np.diag([0.5, 0.5, 0.5]))
        np.testing.assert_allclose(R, np.eye(3))
        self.assertAlmostEqual(scale, 0.5)
        with self.assertRaises(AssertionError):
            hypersim_camera_orientation(np.diag([0.5, 0.6, 0.5]))
        with self.assertRaises(AssertionError):
            hypersim_camera_orientation(np.diag([-1.0, 1.0, 1.0]))

    def test_world_translation_scaled_once_camera_axes_converted_on_right(self):
        T = np.eye(4)
        T[:3, 3] = [100, 200, 300]
        cv = gl_asset_pose_to_cv_meters(T, 0.01)
        np.testing.assert_allclose(cv[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(cv[:3, :3], np.diag([1, -1, -1]))


META = (
    Path(os.environ.get("FYP3D_BASE", "/nonexistent"))
    / "sources/hypersim/contrib/mikeroberts3000/metadata_camera_parameters.csv"
)


@unittest.skipUnless(META.is_file(), "actual publisher camera metadata required")
class HypersimMetadataTests(unittest.TestCase):
    def test_every_publisher_camera_retains_original_ray_directions(self):
        checked = tilted = 0
        for row in csv.DictReader(META.open()):
            K, R, w, h = hypersim_camera_model(row)
            M = np.array(
                [
                    [float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)]
                    for i in range(3)
                ]
            )
            pixels = np.array([[0, 0, 1], [w - 1, h - 1, 1], [w / 2, h / 2, 1.0]])
            uv = (
                pixels
                @ np.array(
                    [[2 / w, 0, 1 / w - 1], [0, -2 / h, 1 - 1 / h], [0, 0, 1.0]]
                ).T
            )
            original = uv @ M.T @ np.diag([1, -1, -1])
            restored = pixels @ np.linalg.inv(K).T @ R
            original /= np.linalg.norm(original, axis=1)[:, None]
            restored /= np.linalg.norm(restored, axis=1)[:, None]
            np.testing.assert_allclose(restored, original, atol=1e-12)
            tilted += not np.allclose(R, np.eye(3), atol=1e-8)
            checked += 1
        self.assertGreater(checked, 400)
        self.assertGreater(tilted, 0)

    def test_float16_rounding_bound_increases_with_stored_magnitude(self):
        # Exact values before storage are retained here to test the interval.
        position = np.array([[[100.013, -84.021, 73.016]]])
        distance = np.linalg.norm(position, axis=-1)
        stored = position.astype(np.float16)
        radial = distance.astype(np.float16)
        reconstructed = position / distance[..., None] * radial[..., None]
        error = np.linalg.norm(reconstructed - stored.astype(float), axis=-1)
        bound = hypersim_quantization_bound(stored, radial, 1.0)
        self.assertTrue((error <= bound).all())
        # A wrong coordinate convention cannot be accepted by storage tolerance.
        wrong = reconstructed.copy()
        wrong[..., 2] *= -1
        self.assertTrue((np.linalg.norm(wrong - stored, axis=-1) > bound).all())

    def test_full_image_rays_match_publisher_matrix_including_half_pixel_centres(self):
        row = next(csv.DictReader(META.open()))
        K, w, h = hypersim_intrinsics(row)
        M = np.array(
            [[float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)] for i in range(3)]
        )
        v, u = np.indices((h, w))
        uv = np.stack(
            [2 * (u + 0.5) / w - 1, 1 - 2 * (v + 0.5) / h, np.ones_like(u)], -1
        )
        rays_gl = uv @ M.T
        rays_cv = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
        transformed = rays_gl @ np.diag([1, -1, -1])
        transformed /= transformed[..., 2, None]
        np.testing.assert_allclose(rays_cv, transformed, atol=1e-12)
        z, valid = radial_distance_to_z(np.full((h, w), 2.0), K)
        self.assertTrue(valid.all())
        self.assertLess(z[0, 0], z[h // 2, w // 2])
        np.testing.assert_allclose(
            np.linalg.norm(rays_cv * z[..., None], axis=-1), 2.0, atol=1e-12
        )

    def test_nonrigid_world_basis_preserves_publisher_rays_with_per_frame_K(self):
        row = next(csv.DictReader(META.open()))
        basis = np.array([[0.8, 0.1, 0], [0, 1.1, 0.03], [0, 0, 0.95]])
        K, T, w, h = hypersim_world_camera_model(row, basis, [100, 200, 300], 0.01)
        np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)
        M = np.array(
            [[float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)] for i in range(3)]
        )
        pixels = np.array([[0, 0, 1.0], [w - 1, h - 1, 1.0], [511, 123, 1.0]])
        uv = (
            pixels
            @ np.array([[2 / w, 0, 1 / w - 1], [0, -2 / h, 1 - 1 / h], [0, 0, 1.0]]).T
        )
        original = uv @ M.T @ basis.T
        restored = pixels @ np.linalg.inv(K).T @ T[:3, :3].T
        original /= np.linalg.norm(original, axis=1)[:, None]
        restored /= np.linalg.norm(restored, axis=1)[:, None]
        np.testing.assert_allclose(original, restored, atol=1e-12)
        # The world camera is rigid; anisotropy is retained in calibrated K.
        self.assertGreater(abs(K[0, 0] - K[1, 1]), 1.0)

    def test_invalid_range_does_not_turn_into_valid_zero_depth(self):
        z, valid = radial_distance_to_z(
            np.array([[np.inf, 0], [np.nan, -1]]), np.eye(3)
        )
        self.assertFalse(valid.any())
        np.testing.assert_array_equal(z, 0)


if __name__ == "__main__":
    unittest.main()
