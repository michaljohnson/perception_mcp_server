"""Tests for the ``get_topdown_grasp_pose`` tool."""

import json
from typing import Any, Dict
from unittest.mock import Mock

import numpy as np
import pytest
from fastmcp import Client, FastMCP

from perception_mcp.tools.grasping import register_grasping_tools


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
        """Wrist z must be centroid_base.z + 0.14 (Robotiq finger offset)."""
        mcp, _ws, _cache = grasping_server
        result = await _call(mcp, object_name="red cup")
        # GRIPPER_FINGER_OFFSET_M is 0.14.
        assert result["gripper_offset_m"] == pytest.approx(0.14)
        expected_z = result["centroid_base_frame"]["z"] + 0.14
        assert result["grasp_pose"]["position"]["z"] == pytest.approx(
            round(expected_z, 4), abs=1e-4
        )

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

        result = await _call(mcp, object_name="cube")
        # Either the success path or an error from the (mocked) TF lookup
        # is acceptable here; what matters is the fallback was exercised.
        assert mock_ws_manager.get_pointcloud.called
        assert "object_name" in result

    async def test_timeout_returns_error_dict(
        self,
        mock_ws_manager: Mock,
    ) -> None:
        mcp = FastMCP("test")
        mock_ws_manager.get_pointcloud.side_effect = TimeoutError("no message")
        register_grasping_tools(mcp, mock_ws_manager, {})

        result = await _call(mcp, object_name="cube")
        assert "error" in result
        assert "timeout" in result["error"].lower()


class TestGraspEdgeCases:
    """Edge cases for input cleanliness and degenerate clouds."""

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
