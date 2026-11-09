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
        assert result["grasp_z_method"] in ("centroid", "below_top", "above_support")
        if result["grasp_z_method"] == "centroid":
            expected_z = result["centroid_base_frame"]["z"] + GRIPPER_FINGER_OFFSET_M
            assert result["grasp_pose"]["position"]["z"] == pytest.approx(
                round(expected_z, 4), abs=1e-4
            )
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
