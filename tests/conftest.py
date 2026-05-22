"""Shared pytest fixtures for perception-mcp-server tests.

Provides mock objects for the WebSocketManager and standard sample data
(point clouds, TF transforms, segmentation cache entries) so individual
test modules stay focused on behaviour rather than setup.
"""

from typing import Any, Dict
from unittest.mock import Mock

import numpy as np
import pytest


# Identity TF transform (camera frame == base frame) — useful as a
# neutral baseline so test expectations match the input data.
IDENTITY_TRANSLATION: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
IDENTITY_ROTATION: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


@pytest.fixture
def identity_translation() -> Dict[str, float]:
    """Identity translation (no shift)."""
    return dict(IDENTITY_TRANSLATION)


@pytest.fixture
def identity_rotation() -> Dict[str, float]:
    """Identity rotation (no orientation change)."""
    return dict(IDENTITY_ROTATION)


@pytest.fixture
def small_point_cloud() -> np.ndarray:
    """A 50-point cloud around (0.5, 0.0, 0.3) in camera frame.

    Elongated along x to give a deterministic PCA principal axis at 0
    radians. Useful for testing centroid + PCA logic without TF effects.
    """
    rng = np.random.default_rng(seed=42)
    n = 50
    cx, cy, cz = 0.5, 0.0, 0.3
    long_axis = rng.normal(loc=cx, scale=0.05, size=n)
    short_axis = rng.normal(loc=cy, scale=0.01, size=n)
    z = rng.normal(loc=cz, scale=0.005, size=n)
    return np.stack([long_axis, short_axis, z], axis=1)


@pytest.fixture
def round_point_cloud() -> np.ndarray:
    """A 100-point cloud with roughly equal xy spread (aspect ratio ≈ 1).

    Triggers the PCA fallback to default top-down orientation.
    """
    rng = np.random.default_rng(seed=7)
    n = 100
    xy = rng.normal(loc=0.0, scale=0.03, size=(n, 2))
    z = rng.normal(loc=0.4, scale=0.005, size=(n, 1))
    return np.hstack([xy, z])


@pytest.fixture
def sample_segmentation_cache(small_point_cloud: np.ndarray) -> Dict[str, Any]:
    """A populated segmentation cache as produced by segment_objects().

    Mirrors the structure expected by grasping.py / placing.py: points
    + colors arrays plus a snapshot of the camera→base TF translation
    and rotation at the moment of segmentation.
    """
    n = small_point_cloud.shape[0]
    return {
        "points": small_point_cloud,
        "colors": np.full((n, 3), 128, dtype=np.uint8),
        "frame_id": "arm_camera_color_optical_frame",
        "tf_translation": {"x": 0.30, "y": 0.0, "z": 1.20},
        "tf_rotation": {"x": -0.5, "y": 0.5, "z": -0.5, "w": 0.5},
        "prompt": "red cup",
        "camera": "arm",
        "timestamp": 1700000000.0,
    }


@pytest.fixture
def mock_ws_manager() -> Mock:
    """A mock WebSocketManager that returns sensible defaults.

    By default `send_action_goal` returns an identity-TF lookup result
    (translation=(0,0,0), rotation=(0,0,0,1)). Individual tests can
    override the return_value or side_effect to exercise other paths.
    """
    ws = Mock()
    ws.send_action_goal.return_value = {
        "transform": {
            "transform": {
                "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        }
    }
    return ws
