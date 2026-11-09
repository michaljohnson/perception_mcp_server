"""Tests for the pure-math helpers in ``perception_mcp.utils.transforms``.

Covers quaternion / rotation matrix conversions, point-frame transforms,
PCA principal-axis extraction, grasp-yaw wrapping, and the composed
top-down quaternion. ROS-dependent helpers (``_tf_lookup``) are tested
against a mocked WebSocketManager.
"""

import math
from typing import Dict
from unittest.mock import Mock

from pathlib import Path

import numpy as np
import pytest

from perception_mcp.utils.transforms import (
    GRIPPER_FINGER_OFFSET_M,
    TOP_DOWN_ORIENTATION,
    _oriented_topdown_quaternion,
    _PCA_ASPECT_RATIO_MIN,
    _grasp_xy,
    _grasp_z,
    _principal_axis_angle_xy,
    _quat_multiply,
    _quat_to_rotation_matrix,
    _shortest_grasp_yaw,
    _tf_lookup,
    _transform_point,
    _transform_points,
)


class TestModuleConstants:
    """Tests for the exported constants."""

    def test_top_down_orientation_is_180_deg_flip_about_x(self) -> None:
        """TOP_DOWN_ORIENTATION must encode a 180-degree rotation about X."""
        assert TOP_DOWN_ORIENTATION == {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}

    def test_gripper_finger_offset_is_robotiq_140(self) -> None:
        """Wrist → middle of the finger pads, as measured on the Summit's Robotiq 2F-140."""
        assert GRIPPER_FINGER_OFFSET_M == pytest.approx(0.188)

    def test_pca_aspect_ratio_threshold_is_sane(self) -> None:
        """The PCA fallback threshold must sit above unit-aspect noise."""
        # Anything between 1.2 and 2.0 is defensible; the absolute floor
        # is "well above 1.0" so segmentation noise on cubes / balls
        # doesn't accidentally trigger an oriented grasp.
        assert 1.0 < _PCA_ASPECT_RATIO_MIN < 2.5


class TestQuatToRotationMatrix:
    """Tests for ``_quat_to_rotation_matrix``."""

    def test_identity_quaternion_yields_identity_matrix(self) -> None:
        R = _quat_to_rotation_matrix(0.0, 0.0, 0.0, 1.0)
        assert np.allclose(R, np.eye(3))

    def test_180_about_x_flips_y_and_z(self) -> None:
        R = _quat_to_rotation_matrix(1.0, 0.0, 0.0, 0.0)
        expected = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        assert np.allclose(R, expected, atol=1e-9)

    def test_90_about_z_swaps_x_and_y_axes(self) -> None:
        R = _quat_to_rotation_matrix(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        # +x in source maps to +y in target.
        assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)
        # +y in source maps to -x in target.
        assert np.allclose(R @ np.array([0.0, 1.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-9)

    def test_rotation_matrix_is_orthogonal(self) -> None:
        """Any unit quaternion must produce an orthonormal rotation matrix."""
        R = _quat_to_rotation_matrix(-0.5, 0.5, -0.5, 0.5)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-9)


class TestTransformPoint:
    """Tests for ``_transform_point``."""

    def test_identity_transform_returns_input(
        self, identity_translation: Dict[str, float],
        identity_rotation: Dict[str, float],
    ) -> None:
        out = _transform_point(
            {"x": 1.5, "y": -2.3, "z": 0.7},
            identity_translation,
            identity_rotation,
        )
        assert out == {"x": 1.5, "y": -2.3, "z": 0.7}

    def test_pure_translation(
        self, identity_rotation: Dict[str, float]
    ) -> None:
        out = _transform_point(
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 1.0, "y": 2.0, "z": 3.0},
            identity_rotation,
        )
        assert out == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_pure_rotation(
        self, identity_translation: Dict[str, float]
    ) -> None:
        # 90 deg about z -> (1, 0, 0) becomes (0, 1, 0)
        rot = {
            "x": 0.0, "y": 0.0,
            "z": math.sin(math.pi / 4),
            "w": math.cos(math.pi / 4),
        }
        out = _transform_point(
            {"x": 1.0, "y": 0.0, "z": 0.0},
            identity_translation,
            rot,
        )
        assert out["x"] == pytest.approx(0.0, abs=1e-4)
        assert out["y"] == pytest.approx(1.0, abs=1e-4)
        assert out["z"] == pytest.approx(0.0, abs=1e-4)

    def test_result_is_rounded_to_4_decimals(
        self, identity_rotation: Dict[str, float]
    ) -> None:
        """The dict output rounds each coordinate to 4 decimals."""
        out = _transform_point(
            {"x": 0.123456789, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": 0.0},
            identity_rotation,
        )
        assert out["x"] == 0.1235  # 4-decimal rounding


class TestTransformPoints:
    """Tests for ``_transform_points`` (vectorised)."""

    def test_identity_transform_returns_input(
        self, identity_translation: Dict[str, float],
        identity_rotation: Dict[str, float],
    ) -> None:
        points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]])
        out = _transform_points(points, identity_translation, identity_rotation)
        assert np.allclose(out, points)

    def test_pure_translation_applies_to_all_rows(
        self, identity_rotation: Dict[str, float]
    ) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        translation = {"x": 5.0, "y": -2.0, "z": 0.5}
        out = _transform_points(points, translation, identity_rotation)
        assert np.allclose(out, np.array([[5.0, -2.0, 0.5], [6.0, -1.0, 1.5]]))

    def test_output_preserves_shape_and_dtype(
        self, identity_translation: Dict[str, float],
        identity_rotation: Dict[str, float],
    ) -> None:
        points = np.random.default_rng(seed=0).normal(size=(20, 3))
        out = _transform_points(points, identity_translation, identity_rotation)
        assert out.shape == (20, 3)
        assert out.dtype == np.float64

    def test_no_rounding_on_vectorised_output(
        self, identity_rotation: Dict[str, float]
    ) -> None:
        """Unlike ``_transform_point``, the vectorised form does NOT round."""
        points = np.array([[0.123456789, 0.0, 0.0]])
        out = _transform_points(points, {"x": 0.0, "y": 0.0, "z": 0.0}, identity_rotation)
        assert out[0, 0] == pytest.approx(0.123456789, abs=1e-12)


class TestPrincipalAxisAngleXy:
    """Tests for ``_principal_axis_angle_xy``."""

    def test_degenerate_input_returns_zero_aspect_one(self) -> None:
        """Fewer than 3 points must return the safe fallback (0, 1)."""
        for n in (0, 1, 2):
            points = np.zeros((n, 3))
            angle, ratio = _principal_axis_angle_xy(points)
            assert angle == 0.0
            assert ratio == 1.0

    def test_elongated_along_x_axis_returns_zero_angle(self) -> None:
        """A cloud spread along base +x should yield angle ≈ 0."""
        rng = np.random.default_rng(seed=42)
        n = 100
        x = rng.normal(loc=0.0, scale=0.5, size=n)
        y = rng.normal(loc=0.0, scale=0.01, size=n)
        z = np.zeros(n)
        points = np.stack([x, y, z], axis=1)
        angle, ratio = _principal_axis_angle_xy(points)
        # Angle wraps to either ≈ 0 or ≈ ±pi (sign of leading eigvec is
        # arbitrary). Either represents the +x long axis.
        wrapped = abs(angle) % math.pi
        assert wrapped < 0.05 or wrapped > math.pi - 0.05
        assert ratio > 10  # very elongated

    def test_elongated_along_y_axis_returns_quarter_pi_or_negative(self) -> None:
        """A cloud spread along base +y should yield ≈ ±pi/2."""
        rng = np.random.default_rng(seed=42)
        n = 100
        x = rng.normal(loc=0.0, scale=0.01, size=n)
        y = rng.normal(loc=0.0, scale=0.5, size=n)
        z = np.zeros(n)
        points = np.stack([x, y, z], axis=1)
        angle, ratio = _principal_axis_angle_xy(points)
        # Eigenvector sign is arbitrary so +pi/2 and -pi/2 are equivalent.
        assert abs(abs(angle) - math.pi / 2) < 0.05
        assert ratio > 10

    def test_round_cloud_returns_aspect_near_one(self) -> None:
        """A roughly symmetric xy spread should yield aspect ≈ 1."""
        rng = np.random.default_rng(seed=7)
        n = 200
        xy = rng.normal(loc=0.0, scale=0.05, size=(n, 2))
        points = np.hstack([xy, np.zeros((n, 1))])
        _, ratio = _principal_axis_angle_xy(points)
        assert 0.9 < ratio < 1.4  # noise-tolerant band

    def test_collapsed_short_axis_returns_safe_fallback(self) -> None:
        """Numerically zero short-axis variance returns the (0, 1) fallback."""
        # All points colinear along x — short-axis variance = 0.
        points = np.array([[i * 0.01, 0.0, 0.0] for i in range(20)])
        angle, ratio = _principal_axis_angle_xy(points)
        assert angle == 0.0
        assert ratio == 1.0


class TestShortestGraspYaw:
    """Tests for ``_shortest_grasp_yaw``."""

    def test_zero_stays_zero(self) -> None:
        assert _shortest_grasp_yaw(0.0) == 0.0

    def test_already_in_range_unchanged(self) -> None:
        for y in (-1.0, -0.5, 0.5, 1.0):
            assert _shortest_grasp_yaw(y) == pytest.approx(y)

    def test_just_above_half_pi_wraps_down_by_pi(self) -> None:
        # pi/2 + 0.1 → 0.1 - pi/2 (i.e. pi/2 + 0.1 - pi).
        assert _shortest_grasp_yaw(math.pi / 2 + 0.1) == pytest.approx(
            0.1 - math.pi / 2, abs=1e-9
        )

    def test_just_below_negative_half_pi_wraps_up_by_pi(self) -> None:
        assert _shortest_grasp_yaw(-math.pi / 2 - 0.1) == pytest.approx(
            -0.1 + math.pi / 2, abs=1e-9
        )

    def test_pi_wraps_to_zero(self) -> None:
        """pi is equivalent to 0 under 180-degree symmetry."""
        assert _shortest_grasp_yaw(math.pi) == pytest.approx(0.0, abs=1e-9)

    def test_minus_pi_wraps_to_zero(self) -> None:
        assert _shortest_grasp_yaw(-math.pi) == pytest.approx(0.0, abs=1e-9)


class TestQuatMultiply:
    """Tests for ``_quat_multiply``."""

    def test_identity_left(self) -> None:
        identity = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        q = {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.4}
        assert _quat_multiply(identity, q) == q

    def test_identity_right(self) -> None:
        identity = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        q = {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.4}
        assert _quat_multiply(q, identity) == q

    def test_two_90deg_z_rotations_produce_180deg_z(self) -> None:
        q90 = {
            "x": 0.0, "y": 0.0,
            "z": math.sin(math.pi / 4),
            "w": math.cos(math.pi / 4),
        }
        out = _quat_multiply(q90, q90)
        # Should equal a 180-deg z rotation = (0, 0, 1, 0)
        assert out["x"] == pytest.approx(0.0, abs=1e-9)
        assert out["y"] == pytest.approx(0.0, abs=1e-9)
        assert out["z"] == pytest.approx(1.0, abs=1e-9)
        assert out["w"] == pytest.approx(0.0, abs=1e-9)


class TestOrientedTopdownQuaternion:
    """Tests for ``_oriented_topdown_quaternion``."""

    def test_zero_yaw_returns_top_down_orientation(self) -> None:
        out = _oriented_topdown_quaternion(0.0)
        for k in ("x", "y", "z", "w"):
            assert out[k] == pytest.approx(TOP_DOWN_ORIENTATION[k], abs=1e-9)

    def test_quarter_turn_yaw_is_unit_quaternion(self) -> None:
        out = _oriented_topdown_quaternion(math.pi / 2)
        norm_sq = sum(v * v for v in out.values())
        assert norm_sq == pytest.approx(1.0, abs=1e-9)

    def test_arbitrary_yaw_preserves_unit_norm(self) -> None:
        for yaw in (-1.2, -0.5, 0.0, 0.7, 1.4):
            out = _oriented_topdown_quaternion(yaw)
            norm_sq = sum(v * v for v in out.values())
            assert norm_sq == pytest.approx(1.0, abs=1e-9)


class TestTfLookup:
    """Tests for ``_tf_lookup``: mocked rosbridge action call."""

    def test_returns_translation_and_rotation_dicts(self) -> None:
        ws = Mock()
        ws.send_action_goal.return_value = {
            "transform": {
                "transform": {
                    "translation": {"x": 1.2, "y": -0.3, "z": 0.5},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.707, "w": 0.707},
                }
            }
        }
        translation, rotation = _tf_lookup(
            ws, source_frame="arm_camera_color_optical_frame"
        )
        assert translation == {"x": 1.2, "y": -0.3, "z": 0.5}
        assert rotation == {"x": 0.0, "y": 0.0, "z": 0.707, "w": 0.707}

    def test_calls_canonical_buffer_server(self) -> None:
        """The lookup must target the canonical-namespace action.

        The whole point of the canonical action is to avoid the race
        against the default-named buffer servers in MoveIt / rviz /
        segmentation nodes.
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
        _tf_lookup(ws, source_frame="some_frame")
        assert ws.send_action_goal.called
        kwargs = ws.send_action_goal.call_args.kwargs
        assert kwargs["action_name"] == "/canonical_tf/tf2_buffer_server"
        assert kwargs["action_type"] == "tf2_msgs/action/LookupTransform"

    def test_passes_target_and_source_frames(self) -> None:
        ws = Mock()
        ws.send_action_goal.return_value = {
            "transform": {
                "transform": {
                    "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
            }
        }
        _tf_lookup(
            ws, source_frame="src_link", target_frame="base_footprint"
        )
        goal = ws.send_action_goal.call_args.kwargs["goal"]
        assert goal["target_frame"] == "base_footprint"
        assert goal["source_frame"] == "src_link"

    def test_propagates_ws_manager_exceptions(self) -> None:
        ws = Mock()
        ws.send_action_goal.side_effect = RuntimeError("rosbridge down")
        with pytest.raises(RuntimeError, match="rosbridge down"):
            _tf_lookup(ws, source_frame="x")


DATA = Path(__file__).parent / "data"
# Arm camera origin in base_footprint at look_forward, from TF on the robot.
LOOK_FORWARD_CAMERA_XY = (0.4314, -0.1109)


def _box_seen_from_front(centre_x: float, size: float, n: int = 40) -> np.ndarray:
    """Front face (x = centre - size/2) and top of a box; the back is not seen."""
    h = size / 2
    s = np.linspace(-h, h, n)
    front = np.array([[centre_x - h, y, z + h] for y in s for z in s])
    top = np.array([[centre_x + x, y, size] for x in s for y in s])
    return np.vstack([front, front, top])  # the near face dominates, as on the robot


class TestGraspXY:
    """Tests for ``_grasp_xy`` (near face + half width along the view)."""

    def test_box_seen_from_front_with_stray_points(self) -> None:
        cloud = _box_seen_from_front(1.0, 0.057)
        stray = np.array([[1.09, 0.0, 0.03]] * (len(cloud) // 50))  # 2 % behind the box
        x, y, method = _grasp_xy(np.vstack([cloud, stray]), (0.0, 0.0))
        assert method == "near_face"
        assert x == pytest.approx(1.0, abs=0.003)
        assert y == pytest.approx(0.0, abs=0.003)

    def test_camera_above_object_uses_trimmed_bbox(self) -> None:
        cloud = _box_seen_from_front(1.0, 0.057)
        _, _, method = _grasp_xy(cloud, (1.0, 0.0))
        assert method == "trimmed_bbox"

    # Real arm-camera clouds (base_footprint); x confirmed by hovering
    # the gripper over the objects (by eye, about +-1 cm) and grasping them.
    @pytest.mark.parametrize("name,x_real", [("cube_1", 1.021), ("cylinder_1", 0.983)])
    def test_real_clouds(self, name: str, x_real: float) -> None:
        x, _, method = _grasp_xy(np.load(DATA / f"{name}.npy"), LOOK_FORWARD_CAMERA_XY)
        assert method == "near_face"
        assert x == pytest.approx(x_real, abs=0.005)


def _column(height: float, n: int = 400) -> np.ndarray:
    """Points of an upright object from 1.5 cm (bottom edge unseen) to its top."""
    z = np.linspace(0.015, height, n)
    return np.stack([np.full(n, 1.0), np.zeros(n), z], axis=1)


class TestGraspZ:
    """Tests for ``_grasp_z`` (pad middle at the centroid, below the top, above the support)."""

    def test_short_object_uses_centroid(self) -> None:
        cloud = _column(0.10)
        grasp_z, pre_z, method = _grasp_z(cloud)
        assert method == "centroid"
        assert grasp_z == pytest.approx(cloud[:, 2].mean() + 0.188, abs=1e-6)
        assert pre_z == pytest.approx(np.percentile(cloud[:, 2], 99) + 0.225 + 0.10, abs=1e-6)

    def test_tall_object_pads_at_most_5_cm_below_the_top(self) -> None:
        cloud = _column(0.245)
        grasp_z, _, method = _grasp_z(cloud)
        assert method == "below_top"
        assert grasp_z - 0.188 == pytest.approx(np.percentile(cloud[:, 2], 99) - 0.05, abs=1e-6)

    def test_flat_object_keeps_fingertips_above_its_bottom(self) -> None:
        cloud = _column(0.02)
        grasp_z, _, method = _grasp_z(cloud)
        assert method == "above_support"
        assert grasp_z - 0.225 == pytest.approx(np.percentile(cloud[:, 2], 1) - 0.005, abs=1e-6)

    # 24 cm bottle: grasped on the robot at flange 0.37 (6.3 cm above the pads, no palm
    # contact); the centroid rule gave 0.3227 (11 cm above the pads, into the palm).
    def test_real_bottle(self) -> None:
        grasp_z, pre_z, method = _grasp_z(np.load(DATA / "bottle_1.npy"))
        assert method == "below_top"
        assert 0.37 <= grasp_z <= 0.40
        assert pre_z == pytest.approx(0.245 + 0.225 + 0.10, abs=0.01)


class TestPCAOnRealClouds:
    """The aspect threshold separates the real robot clouds."""

    @pytest.mark.parametrize("name", ["cube_1", "cylinder_1", "bottle_1"])
    def test_round_and_square_objects_not_oriented(self, name: str) -> None:
        _, ratio = _principal_axis_angle_xy(np.load(DATA / f"{name}.npy"))
        assert ratio < _PCA_ASPECT_RATIO_MIN

    def test_may_cube_noise_not_oriented(self) -> None:
        # Cube clouds reach aspect 1.21-1.30 from noise; 1.4 must stay above that.
        rng = np.random.default_rng(0)
        xy = rng.uniform([-0.030, -0.039], [0.030, 0.039], size=(2000, 2))  # ratio 1.30
        cloud = np.column_stack([xy + [1.0, 0.0], np.full(2000, 0.05)])
        _, ratio = _principal_axis_angle_xy(cloud)
        assert 1.25 < ratio < _PCA_ASPECT_RATIO_MIN

    def test_uno_deck_oriented_across_its_short_side(self) -> None:
        # Deck at about 51 deg; picked on the robot with grasp yaw -39.4 deg.
        angle, ratio = _principal_axis_angle_xy(np.load(DATA / "uno_1.npy"))
        assert ratio >= _PCA_ASPECT_RATIO_MIN
        assert math.degrees(_shortest_grasp_yaw(angle + math.pi / 2)) == pytest.approx(-39.4, abs=2)
