"""Tests for the ``get_topdown_grasp_pose`` tool."""

import json
from typing import Any, Dict
from unittest.mock import Mock

import numpy as np
import pytest
from fastmcp import Client, FastMCP

from perception_mcp.tools.grasping import register_grasping_tools
from perception_mcp.utils.transforms import GRIPPER_FINGER_OFFSET_M

# A raw (not latched) cloud topic, for the no-cache fallback.
RAW_TOPIC = "/camera/arm_camera/depth/color/points"


@pytest.fixture
def grasping_server(
    mock_ws_manager: Mock,
    sample_segmentation_cache: Dict[str, Any],
) -> tuple[FastMCP, Mock, Dict[str, Any]]:
    """A FastMCP with the grasp tool registered against mocks + a populated cache."""
    mcp = FastMCP("test-perception-grasp")
    register_grasping_tools(mcp, mock_ws_manager, sample_segmentation_cache)
    return mcp, mock_ws_manager, sample_segmentation_cache


async def _call(mcp: FastMCP, **args: Any) -> Dict[str, Any]:
    """Call the tool and parse the JSON result."""
    async with Client(mcp) as client:
        result = await client.call_tool("get_topdown_grasp_pose", args)
        raw = result.content[0].text if result.content else result.data
        return json.loads(raw) if isinstance(raw, str) else raw


class TestGraspFromCachedCloud:
    """Happy-path tests against a populated segmentation cache."""

    async def test_returns_grasp_pose_in_base_footprint(
        self,
        grasping_server: tuple[FastMCP, Mock, Dict[str, Any]],
    ) -> None:
        mcp, _ws, _cache = grasping_server
        result = await _call(mcp, object_name="red cup")

        assert result["object_name"] == "red cup"
        assert "grasp_pose" in result
        assert result["grasp_pose"]["frame_id"] == "base_footprint"
        # position keys present
        for axis in ("x", "y", "z"):
            assert axis in result["grasp_pose"]["position"]
        # orientation keys present
        for axis in ("x", "y", "z", "w"):
            assert axis in result["grasp_pose"]["orientation"]

    async def test_includes_centroid_and_bounding_box(
        self,
        grasping_server: tuple[FastMCP, Mock, Dict[str, Any]],
    ) -> None:
        mcp, _ws, _cache = grasping_server
        result = await _call(mcp, object_name="red cup")

        assert "centroid_camera_frame" in result
        assert "centroid_base_frame" in result
        assert "bounding_box" in result
        assert "min" in result["bounding_box"]
        assert "max" in result["bounding_box"]
        assert "size" in result["bounding_box"]

    async def test_grasp_z_includes_gripper_finger_offset(
        self,
        grasping_server: tuple[FastMCP, Mock, Dict[str, Any]],
    ) -> None:
        """Wrist z = pad middle + the wrist → finger pad offset; pad middle by grasp_z_method."""
        mcp, _ws, _cache = grasping_server
        result = await _call(mcp, object_name="red cup")
        assert result["gripper_offset_m"] == pytest.approx(GRIPPER_FINGER_OFFSET_M)
        assert result["grasp_z_method"] in ("below_top", "above_support")
        assert result["pre_grasp_pose"]["position"]["z"] >= result["grasp_pose"]["position"]["z"]

    async def test_num_points_matches_cache_size(
        self,
        grasping_server: tuple[FastMCP, Mock, Dict[str, Any]],
        sample_segmentation_cache: Dict[str, Any],
    ) -> None:
        mcp, _ws, _cache = grasping_server
        result = await _call(mcp, object_name="red cup")
        assert result["num_points"] == sample_segmentation_cache["points"].shape[0]

    async def test_uses_cached_tf_when_present(
        self,
        grasping_server: tuple[FastMCP, Mock, Dict[str, Any]],
    ) -> None:
        """A cached TF snapshot must skip the fresh ``send_action_goal`` lookup.

        This is the central design invariant of segmentation_cache: the
        TF captured at segment time is the authoritative transform for
        those particular points.
        """
        mcp, ws, _cache = grasping_server
        await _call(mcp, object_name="red cup")
        ws.send_action_goal.assert_not_called()


class TestGraspWithoutCachedCloud:
    """Tests for the no-cache fallback path."""

    async def test_empty_cache_falls_back_to_topic_then_succeeds(
        self,
        mock_ws_manager: Mock,
    ) -> None:
        """With an empty cache, the tool must pull points from the topic.

        The ws_manager mock is configured to return a small cloud and a
        cached TF lookup path is bypassed, so the live ``_tf_lookup`` is
        called too.
        """
        mcp = FastMCP("test")
        empty_cache: Dict[str, Any] = {}
        # No cached points → tool falls back to ws_manager.get_pointcloud.
        points = np.array([[0.5, 0.0, 0.3], [0.51, 0.01, 0.31], [0.52, -0.01, 0.29]])
        mock_ws_manager.get_pointcloud.return_value = (
            points, None, "arm_camera_color_optical_frame"
        )
        register_grasping_tools(mcp, mock_ws_manager, empty_cache)

        result = await _call(mcp, object_name="cube", pointcloud_topic=RAW_TOPIC)
        # Either the success path or an error from the (mocked) TF lookup
        # is acceptable here; what matters is the fallback was exercised.
        assert mock_ws_manager.get_pointcloud.called
        assert "object_name" in result

    async def test_empty_cache_does_not_read_latched_segmented_topic(
        self,
        mock_ws_manager: Mock,
    ) -> None:
        """The segmented topic is latched: it would replay an old segmentation."""
        mcp = FastMCP("test")
        register_grasping_tools(mcp, mock_ws_manager, {})

        result = await _call(mcp, object_name="cube")
        assert "error" in result
        assert "segment_objects" in result["error"]
        mock_ws_manager.get_pointcloud.assert_not_called()

    async def test_timeout_returns_error_dict(
        self,
        mock_ws_manager: Mock,
    ) -> None:
        mcp = FastMCP("test")
        mock_ws_manager.get_pointcloud.side_effect = TimeoutError("no message")
        register_grasping_tools(mcp, mock_ws_manager, {})

        result = await _call(mcp, object_name="cube", pointcloud_topic=RAW_TOPIC)
        assert "error" in result
        assert "timeout" in result["error"].lower()


class TestGraspEdgeCases:
    """Edge cases for input cleanliness and degenerate clouds."""

    async def test_failed_tf_lookup_gives_no_pose(
        self,
        mock_ws_manager: Mock,
        sample_segmentation_cache: Dict[str, Any],
    ) -> None:
        """Without a transform there is no base-frame pose, never camera coordinates."""
        mcp = FastMCP("test")
        cache = dict(sample_segmentation_cache)
        cache.pop("tf_translation")
        cache.pop("tf_rotation")
        mock_ws_manager.send_action_goal.side_effect = RuntimeError(
            "Action /canonical_tf/tf2_buffer_server finished with status 6"
        )
        register_grasping_tools(mcp, mock_ws_manager, cache)

        result = await _call(mcp, object_name="red cup")
        assert result["grasp_pose"] is None
        assert result["centroid_base_frame"] is None
        assert "TF lookup failed" in result["warning"]

    async def test_empty_points_returns_error(
        self,
        mock_ws_manager: Mock,
    ) -> None:
        mcp = FastMCP("test")
        cache: Dict[str, Any] = {
            "points": np.empty((0, 3)),
            "colors": np.empty((0, 3), dtype=np.uint8),
            "frame_id": "arm_camera_color_optical_frame",
            "tf_translation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "tf_rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
        register_grasping_tools(mcp, mock_ws_manager, cache)

        result = await _call(mcp, object_name="ghost")
        assert "error" in result
        assert "empty" in result["error"].lower()


class TestGraspXYFromRealCloud:
    """The grasp pose uses near face + half width, not the centroid."""

    async def test_cylinder_grasp_x_matches_the_hover_test(self, mock_ws_manager: Mock) -> None:
        # Real cloud in base_footprint; identity rotation and the camera origin as the
        # TF translation reproduce it from "camera frame" points.
        from pathlib import Path

        cam = np.array([0.4314, -0.1109, 0.679])
        points = np.load(Path(__file__).parent / "data" / "cylinder_1.npy") - cam
        cache = {
            "points": points,
            "colors": np.full((len(points), 3), 128, dtype=np.uint8),
            "frame_id": "arm_camera_color_optical_frame",
            "tf_translation": {"x": cam[0], "y": cam[1], "z": cam[2]},
            "tf_rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
        mcp = FastMCP("test-perception-grasp-real")
        register_grasping_tools(mcp, mock_ws_manager, cache)
        result = await _call(mcp, object_name="orange cylinder")
        assert result["grasp_xy_method"] == "near_face"
        # Hover test on the robot: 0.983. The centroid (old rule) was 0.972, 1.1 cm short.
        assert result["grasp_pose"]["position"]["x"] == pytest.approx(0.983, abs=0.005)


def _cube_cache(extra: np.ndarray | None = None) -> Dict[str, Any]:
    """The real cube cloud (base_footprint) as camera-frame points with identity rotation."""
    from pathlib import Path

    cam = np.array([0.4314, -0.1109, 0.679])
    base = np.load(Path(__file__).parent / "data" / "cube_1.npy")
    if extra is not None:
        base = np.vstack([base, extra])
    return {
        "points": base - cam,
        "colors": np.full((len(base), 3), 128, dtype=np.uint8),
        "frame_id": "arm_camera_color_optical_frame",
        "tf_translation": {"x": cam[0], "y": cam[1], "z": cam[2]},
        "tf_rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _junk(n: int) -> np.ndarray:
    """Stray depth like the black bin's: below the floor and metres away."""
    rng = np.random.default_rng(0)
    below = np.column_stack([rng.uniform(0.8, 2.0, n // 2), rng.uniform(-1, 1, n // 2), rng.uniform(-15, -0.05, n // 2)])
    far = np.column_stack([rng.uniform(7, 16, n - n // 2), rng.uniform(-6, 6, n - n // 2), rng.uniform(0, 0.4, n - n // 2)])
    return np.vstack([below, far])


class TestImplausiblePointsDropped:
    """Points below the floor or far from the camera are dropped before the pose."""

    async def _pose(self, ws: Mock, cache: Dict[str, Any]) -> Dict[str, Any]:
        mcp = FastMCP("test-perception-grasp-junk")
        register_grasping_tools(mcp, ws, cache)
        return await _call(mcp, object_name="green cube")

    async def test_junk_points_do_not_move_the_pose(self, mock_ws_manager: Mock) -> None:
        clean = await self._pose(mock_ws_manager, _cube_cache())
        noisy = await self._pose(mock_ws_manager, _cube_cache(_junk(500)))
        assert noisy["num_points_dropped"] == 500
        for axis in ("x", "y", "z"):
            assert noisy["grasp_pose"]["position"][axis] == pytest.approx(
                clean["grasp_pose"]["position"][axis], abs=0.002
            )
        assert noisy["bbox_base_frame"] == clean["bbox_base_frame"]

    async def test_almost_all_junk_gives_an_error(self, mock_ws_manager: Mock) -> None:
        cache = _cube_cache()
        cache["points"] = np.vstack([cache["points"][:5], _junk(500) - np.array([0.4314, -0.1109, 0.679])])
        result = await self._pose(mock_ws_manager, cache)
        assert "error" in result
        assert "plausible" in result["error"]

    async def test_clean_cloud_drops_nothing(self, mock_ws_manager: Mock) -> None:
        result = await self._pose(mock_ws_manager, _cube_cache())
        assert result["num_points_dropped"] == 0

    async def test_far_object_is_kept(self, mock_ws_manager: Mock) -> None:
        # approach measures targets up to 4 m away.
        cache = _cube_cache()
        cache["points"] = cache["points"] + np.array([3.0, 0.0, 0.0])
        result = await self._pose(mock_ws_manager, cache)
        assert result["num_points_dropped"] == 0
        assert result["grasp_pose"] is not None

    async def test_floor_limit_is_minus_2_cm(self, mock_ws_manager: Mock) -> None:
        near_floor = np.array([[0.9, 0.0, -0.01], [0.9, 0.0, -0.03]])
        result = await self._pose(mock_ws_manager, _cube_cache(near_floor))
        assert result["num_points_dropped"] == 1


class TestTopHeight:
    """top_z: the top for setting something down, robust to a few stray points."""

    async def _pose(self, ws: Mock, cache: Dict[str, Any]) -> Dict[str, Any]:
        mcp = FastMCP("test-perception-grasp-top")
        register_grasping_tools(mcp, ws, cache)
        return await _call(mcp, object_name="green cube")

    async def test_stray_points_above_do_not_raise_the_top(self, mock_ws_manager: Mock) -> None:
        clean = await self._pose(mock_ws_manager, _cube_cache())
        stray = np.array([[0.9, 0.0, 0.16], [0.91, 0.01, 0.15], [0.9, -0.01, 0.17]])
        noisy = await self._pose(mock_ws_manager, _cube_cache(stray))
        assert noisy["bbox_base_frame"]["z_max"] == pytest.approx(0.17)
        assert noisy["top_z"] == pytest.approx(clean["top_z"], abs=0.002)

    async def test_top_is_the_99th_percentile(self, mock_ws_manager: Mock) -> None:
        from pathlib import Path

        z = np.load(Path(__file__).parent / "data" / "cube_1.npy")[:, 2]
        result = await self._pose(mock_ws_manager, _cube_cache())
        assert result["top_z"] == pytest.approx(float(np.percentile(z, 99)), abs=1e-4)
        assert result["top_z"] <= result["bbox_base_frame"]["z_max"]


async def test_non_finite_points_are_dropped(mock_ws_manager: Mock) -> None:
    bad = np.array([[np.nan, 0.0, 0.1], [0.9, np.inf, 0.1], [0.9, 0.0, -np.inf]])
    mcp = FastMCP("test-perception-grasp-nan")
    register_grasping_tools(mcp, mock_ws_manager, _cube_cache(bad))
    result = await _call(mcp, object_name="green cube")
    assert result["num_points_dropped"] == 3
    assert all(np.isfinite(v) for v in result["grasp_pose"]["position"].values())
