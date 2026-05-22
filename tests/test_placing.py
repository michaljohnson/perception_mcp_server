"""Tests for the ``get_topdown_placing_pose`` tool."""

import json
from typing import Any, Dict
from unittest.mock import Mock

import numpy as np
import pytest
from fastmcp import Client, FastMCP

from perception_mcp.tools.placing import register_placing_tools


@pytest.fixture
def placing_server(
    mock_ws_manager: Mock,
    sample_segmentation_cache: Dict[str, Any],
) -> tuple[FastMCP, Mock, Dict[str, Any]]:
    """A FastMCP with the place tool registered against mocks + a populated cache."""
    mcp = FastMCP("test-perception-place")
    register_placing_tools(mcp, mock_ws_manager, sample_segmentation_cache)
    return mcp, mock_ws_manager, sample_segmentation_cache


async def _call(mcp: FastMCP, **args: Any) -> Dict[str, Any]:
    async with Client(mcp) as client:
        result = await client.call_tool("get_topdown_placing_pose", args)
        raw = result.content[0].text if result.content else result.data
        return json.loads(raw) if isinstance(raw, str) else raw


class TestContainerMode:
    """Container mode: ``object_height_m=0`` (default)."""

    async def test_container_returns_top_down_orientation(
        self,
        placing_server: tuple[FastMCP, Mock, Dict[str, Any]],
    ) -> None:
        mcp, _ws, _cache = placing_server
        result = await _call(
            mcp, object_name="trash bin", top_clearance_m=0.35
        )

        assert "place_pose" in result
        assert result["place_pose"]["frame_id"] == "base_footprint"
        # Top-down orientation: (1, 0, 0, 0)
        assert result["place_pose"]["orientation"] == {
            "x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0
        }

    async def test_container_place_z_is_top_plus_clearance(
        self,
        placing_server: tuple[FastMCP, Mock, Dict[str, Any]],
    ) -> None:
        """wrist_z = top_z + top_clearance_m in container mode."""
        mcp, _ws, _cache = placing_server
        result = await _call(
            mcp, object_name="bin", top_clearance_m=0.35
        )

        # surface_height_m is top_z (the p95 of point cloud z)
        top_z = result["surface_height_m"]
        assert result["place_pose"]["position"]["z"] == pytest.approx(
            top_z + 0.35, abs=1e-3
        )

    async def test_container_reports_clearance_used(
        self,
        placing_server: tuple[FastMCP, Mock, Dict[str, Any]],
    ) -> None:
        mcp, _ws, _cache = placing_server
        result = await _call(
            mcp, object_name="bin", top_clearance_m=0.35
        )
        assert result["top_clearance_m"] == pytest.approx(0.35)


class TestSurfaceMode:
    """Surface mode: ``object_height_m > 0`` adds finger + object height."""

    async def test_surface_mode_lifts_wrist_for_finger_and_object(
        self,
        placing_server: tuple[FastMCP, Mock, Dict[str, Any]],
    ) -> None:
        """wrist_z = top_z + 0.14 (finger) + object_height + clearance."""
        mcp, _ws, _cache = placing_server
        clearance = 0.05
        obj_h = 0.12  # coke can height
        result = await _call(
            mcp,
            object_name="coffee table",
            top_clearance_m=clearance,
            object_height_m=obj_h,
        )

        top_z = result["surface_height_m"]
        # 0.14 is the Robotiq 140 finger offset baked into the tool.
        expected = top_z + 0.14 + obj_h + clearance
        assert result["place_pose"]["position"]["z"] == pytest.approx(
            expected, abs=1e-3
        )

    async def test_x_bias_pushes_centroid_forward(
        self,
        placing_server: tuple[FastMCP, Mock, Dict[str, Any]],
    ) -> None:
        """x_bias_m adds to the centroid x in base frame."""
        mcp, _ws, _cache = placing_server
        result_a = await _call(
            mcp, object_name="t", object_height_m=0.1, x_bias_m=0.0
        )
        result_b = await _call(
            mcp, object_name="t", object_height_m=0.1, x_bias_m=0.15
        )
        x_a = result_a["place_pose"]["position"]["x"]
        x_b = result_b["place_pose"]["position"]["x"]
        assert x_b - x_a == pytest.approx(0.15, abs=1e-3)


class TestPlaceCommonOutputs:
    """Output-shape invariants common to both modes."""

    async def test_response_includes_method_and_diagnostics(
        self,
        placing_server: tuple[FastMCP, Mock, Dict[str, Any]],
    ) -> None:
        mcp, _ws, _cache = placing_server
        result = await _call(mcp, object_name="x", top_clearance_m=0.20)

        for key in (
            "surface_height_m", "surface_centroid", "place_pose",
            "top_clearance_m", "num_points_used",
        ):
            assert key in result, f"missing key: {key}"

    async def test_num_points_used_matches_cache_size(
        self,
        placing_server: tuple[FastMCP, Mock, Dict[str, Any]],
        sample_segmentation_cache: Dict[str, Any],
    ) -> None:
        mcp, _ws, _cache = placing_server
        result = await _call(mcp, object_name="x", top_clearance_m=0.20)
        assert result["num_points_used"] == (
            sample_segmentation_cache["points"].shape[0]
        )


class TestPlaceCacheFallback:
    """No-cache path: tool must pull from the topic and crop on request."""

    async def test_use_cached_false_pulls_from_topic(
        self,
        mock_ws_manager: Mock,
    ) -> None:
        mcp = FastMCP("test")
        empty_cache: Dict[str, Any] = {}
        points = np.array([
            [0.5, 0.0, 0.3], [0.51, 0.01, 0.31], [0.52, -0.01, 0.29],
        ])
        mock_ws_manager.get_pointcloud.return_value = (
            points, None, "arm_camera_color_optical_frame"
        )
        register_placing_tools(mcp, mock_ws_manager, empty_cache)

        await _call(
            mcp,
            object_name="bin",
            use_cached=False,
            crop_center_x=0.5,
            crop_center_y=0.0,
        )
        assert mock_ws_manager.get_pointcloud.called

    async def test_timeout_returns_error_dict(
        self,
        mock_ws_manager: Mock,
    ) -> None:
        mcp = FastMCP("test")
        mock_ws_manager.get_pointcloud.side_effect = TimeoutError("no msg")
        register_placing_tools(mcp, mock_ws_manager, {})

        result = await _call(mcp, object_name="bin", use_cached=False)
        assert "error" in result
