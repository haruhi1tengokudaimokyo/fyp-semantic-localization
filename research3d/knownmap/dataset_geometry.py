"""Explicit dataset units, pixel centres, depth conventions and unknown labels."""

import numpy as np


def hypersim_camera_model(metadata):
    """Factor publisher rays, retaining physical-camera tilt in the camera pose.

    R maps the original OpenCV camera frame to the calibrated pinhole frame.
    Thus T_world_pinhole = T_world_cv @ diag(R.T, 1). Pixel centres follow
    the publisher's 01_casting_rays_that_match_hypersim_images notebook.
    """
    from scipy.linalg import rq

    width = int(float(metadata["settings_output_img_width"]))
    height = int(float(metadata["settings_output_img_height"]))
    M = np.array(
        [[float(metadata[f"M_cam_from_uv_{i}{j}"]) for j in range(3)] for i in range(3)]
    )
    A = np.array(
        [[2 / width, 0, 1 / width - 1], [0, -2 / height, 1 - 1 / height], [0, 0, 1.0]]
    )
    B = np.diag([1.0, -1.0, -1.0]) @ M @ A
    assert np.isfinite(B).all() and np.linalg.det(B) > 0
    K, R = rq(np.linalg.inv(B))
    signs = np.diag(np.sign(np.diag(K)))
    K, R = K @ signs, signs @ R
    K /= K[2, 2]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
    assert np.allclose(K[2], [0, 0, 1], atol=1e-12)
    assert np.all(np.diag(K) > 0)
    # Enforce equality of directions, not an arbitrary scale of the ray matrix.
    restored = R.T @ np.linalg.inv(K)
    assert np.allclose(
        restored / np.linalg.norm(restored), B / np.linalg.norm(B), atol=1e-12
    )
    return K, R, width, height


def rounding_half_interval(values):
    """Conservative half-ULP bound for IEEE round-to-nearest stored floats."""
    values = np.asarray(values)
    assert np.issubdtype(values.dtype, np.floating)
    with np.errstate(over="ignore", invalid="ignore"):
        upper = np.nextafter(values, np.inf, dtype=values.dtype).astype(float)
        lower = np.nextafter(values, -np.inf, dtype=values.dtype).astype(float)
    center = values.astype(float)
    up = np.abs(upper - center)
    down = np.abs(center - lower)
    # At finite format extrema the next finite bin has the same spacing.
    up = np.where(np.isinf(up), down, up)
    down = np.where(np.isinf(down), up, down)
    return 0.5 * np.maximum(up, down)


def hypersim_camera_orientation(matrix):
    """Remove only a positive uniform scale from publisher camera axes.

    Released keyframes can contain axes scaled by 0.5. Ray normalization makes
    this scale unobservable. Reject shear/anisotropic scale; never fit a rotation
    to GT/image results or discard a frame to make an experiment pass.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    assert matrix.shape == (3, 3) and np.isfinite(matrix).all()
    determinant = np.linalg.det(matrix)
    assert determinant > 0
    scale = float(np.cbrt(determinant))
    rotation = matrix / scale
    assert np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0
    ), "Nonuniform or sheared camera axes require explicit investigation"
    return rotation, scale


def hypersim_world_camera_model(
    metadata, source_orientation, source_position, meters_per_asset_unit
):
    """Calibrate the full publisher world-ray matrix without assuming rigid axes.

    This is an exact projective factorization, with no measured image points or
    GT fitting. A nonrigid source basis requires per-frame K, not SVD projection
    onto an assumed rotation. Additional real geometry checks must qualify it.
    """
    from scipy.linalg import rq

    width = int(float(metadata["settings_output_img_width"]))
    height = int(float(metadata["settings_output_img_height"]))
    M = np.array(
        [[float(metadata[f"M_cam_from_uv_{i}{j}"]) for j in range(3)] for i in range(3)]
    )
    A = np.array(
        [[2 / width, 0, 1 / width - 1], [0, -2 / height, 1 - 1 / height], [0, 0, 1.0]]
    )
    basis = np.asarray(source_orientation, dtype=np.float64)
    assert (
        basis.shape == (3, 3) and np.isfinite(basis).all() and np.linalg.det(basis) > 0
    )
    W = basis @ M @ A
    K, R = rq(np.linalg.inv(W))
    signs = np.diag(np.sign(np.diag(K)))
    K, R = K @ signs, signs @ R
    K /= K[2, 2]
    assert np.isclose(np.linalg.det(R), 1, atol=1e-12)
    restored = R.T @ np.linalg.inv(K)
    assert np.allclose(
        restored / np.linalg.norm(restored), W / np.linalg.norm(W), atol=1e-12
    )
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = np.asarray(source_position, dtype=np.float64) * float(
        meters_per_asset_unit
    )
    assert np.isfinite(T).all() and np.all(np.diag(K) > 0)
    return K, T, width, height


def hypersim_quantization_bound(position_raw, radial_raw, meters_per_asset_unit):
    """World-position vector rounding + independent radial-distance rounding.

    Publisher generate_hdf5_from_vrimg.py writes both arrays as float16.
    No localization errors or held-out statistics enter this analytic bound.
    """
    position_bound = np.linalg.norm(rounding_half_interval(position_raw), axis=-1)
    return (
        position_bound * float(meters_per_asset_unit)
        + rounding_half_interval(radial_raw)
        + 1e-6
    )


def hypersim_intrinsics(metadata):
    width = int(float(metadata["settings_output_img_width"]))
    height = int(float(metadata["settings_output_img_height"]))
    M = np.array(
        [[float(metadata[f"M_cam_from_uv_{i}{j}"]) for j in range(3)] for i in range(3)]
    )
    # Hypersim UV is [-1,+1], pixel centres are u+.5/v+.5, and camera is OpenGL.
    uv_from_pixel = np.array(
        [[2 / width, 0, 1 / width - 1], [0, -2 / height, 1 - 1 / height], [0, 0, 1.0]]
    )
    cv_ray_from_pixel = np.diag([1, -1, -1]) @ M @ uv_from_pixel
    K = np.linalg.inv(cv_ray_from_pixel)
    K /= K[2, 2]
    assert np.allclose(K[2], [0, 0, 1]) and K[0, 0] > 0 and K[1, 1] > 0
    return K, width, height


def radial_distance_to_z(distance, K):
    distance = np.asarray(distance, float)
    K = np.asarray(K, float)
    assert distance.ndim == 2 and K.shape == (3, 3) and np.isfinite(K).all()
    v, u = np.indices(distance.shape)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    rays /= rays[..., 2, None]
    valid = np.isfinite(distance) & (distance > 0)
    result = np.zeros_like(distance)
    result[valid] = distance[valid] / np.linalg.norm(rays[valid], axis=1)
    return result, valid


def gl_asset_pose_to_cv_meters(pose, meters_per_asset_unit):
    T = np.array(pose, float, copy=True)
    scale = float(meters_per_asset_unit)
    assert (
        T.shape == (4, 4) and np.isfinite(T).all() and np.isfinite(scale) and scale > 0
    )
    assert np.allclose(T[3], [0, 0, 0, 1])
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5)
    T[:3, 3] *= scale
    return T @ np.diag([1, -1, -1, 1])


def reduce_semantic_labels(raw, dataset):
    raw = np.asarray(raw)
    assert np.issubdtype(raw.dtype, np.integer)
    classes = 150 if dataset == "ade20k" else 40 if dataset == "hypersim" else None
    assert classes is not None
    ignore = (raw == 0) | (raw == 255) if dataset == "ade20k" else raw == -1
    assert np.all(
        ignore | ((raw >= 1) & (raw <= classes))
    ), "Undocumented label IDs; do not silently remap"
    labels = np.full(raw.shape, 255, np.uint8)
    labels[~ignore] = (raw[~ignore] - 1).astype(np.uint8)
    return labels
