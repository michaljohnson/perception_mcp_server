"""Tests for the ``look`` camera-image tool."""

from typing import Any, Dict
from unittest.mock import Mock

import pytest
from fastmcp import Client, FastMCP

from perception_mcp.tools.detection import register_detection_tools


CAMERA_TOPICS: Dict[str, Dict[str, str]] = {
    "front": {"rgb": "/front_rgbd_camera/color/image_raw/compressed"},
    "arm": {"rgb": "/arm_camera/color/image_raw/compressed"},
}

FAKE_JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg-data\xff\xd9"


@pytest.fixture
def detection_server() -> tuple[FastMCP, Mock]:
    """A FastMCP server with the look tool registered against a mocked ws.

    Returns (mcp_instance, ws_mock) so tests can both call the tool via
    a Client and inspect mock call arguments.
    """
    mcp = FastMCP("test-perception-look")
    ws = Mock()
    ws.get_compressed_image.return_value = FAKE_JPEG_BYTES
    register_detection_tools(mcp, ws, CAMERA_TOPICS)
    return mcp, ws


class TestLookCameraSelection:
    """Tests for the camera-argument dispatch."""

    async def test_look_front_returns_single_image(
        self, detection_server: tuple[FastMCP, Mock]
    ) -> None:
        mcp, ws = detection_server
        async with Client(mcp) as client:
            result = await client.call_tool("look", {"camera": "front"})
            assert result.content
            # Exactly one fetch was performed, on the front topic.
            ws.get_compressed_image.assert_called_once_with(
                CAMERA_TOPICS["front"]["rgb"], timeout=10.0
            )

    async def test_look_arm_returns_single_image(
        self, detection_server: tuple[FastMCP, Mock]
    ) -> None:
        mcp, ws = detection_server
        async with Client(mcp) as client:
            result = await client.call_tool("look", {"camera": "arm"})
            assert result.content
            ws.get_compressed_image.assert_called_once_with(
                CAMERA_TOPICS["arm"]["rgb"], timeout=10.0
            )

    async def test_look_both_calls_ws_for_each_camera(
        self, detection_server: tuple[FastMCP, Mock]
    ) -> None:
        mcp, ws = detection_server
        async with Client(mcp) as client:
            result = await client.call_tool("look", {"camera": "both"})
            assert result.content
            assert ws.get_compressed_image.call_count == 2
            called_topics = [
                call.args[0] for call in ws.get_compressed_image.call_args_list
            ]
            assert called_topics == [
                CAMERA_TOPICS["front"]["rgb"],
                CAMERA_TOPICS["arm"]["rgb"],
            ]

    async def test_look_default_camera_is_front(
        self, detection_server: tuple[FastMCP, Mock]
    ) -> None:
        mcp, ws = detection_server
        async with Client(mcp) as client:
            await client.call_tool("look", {})
            ws.get_compressed_image.assert_called_once_with(
                CAMERA_TOPICS["front"]["rgb"], timeout=10.0
            )


class TestLookInvalidCamera:
    """Invalid camera value must raise rather than guess a default."""

    async def test_unknown_camera_raises(
        self, detection_server: tuple[FastMCP, Mock]
    ) -> None:
        mcp, _ws = detection_server
        async with Client(mcp) as client:
            with pytest.raises(Exception):
                await client.call_tool("look", {"camera": "rear"})


class TestLookErrorPropagation:
    """Failures in the websocket layer must surface to the caller."""

    async def test_ws_manager_exception_propagates(
        self, detection_server: tuple[FastMCP, Mock]
    ) -> None:
        mcp, ws = detection_server
        ws.get_compressed_image.side_effect = TimeoutError("camera offline")
        async with Client(mcp) as client:
            with pytest.raises(Exception):
                await client.call_tool("look", {"camera": "front"})
