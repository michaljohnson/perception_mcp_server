"""Tests for ``segment_objects`` against a fake rosbridge.

The segmentation node publishes /segmentation_status and /segmented_pointcloud
latched, so every new subscriber first receives the previous run's result.
The fake bridge replays that latched message on subscribe and delivers the new
run's messages only after the prompt was published.
"""

import json
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import numpy as np
import pytest
import websocket
from fastmcp import Client, FastMCP

from perception_mcp.tools.segmentation import register_segmentation_tools

STATUS = "/segmentation_status"
CLOUD = "/segmented_pointcloud"


def _cloud(stamp_sec: int, marker: float) -> Dict[str, Any]:
    return {"header": {"stamp": {"sec": stamp_sec, "nanosec": 0}}, "marker": marker}


class FakeBridge:
    """Latched messages per topic, and the messages of the run the prompt starts."""

    def __init__(
        self,
        latched: Dict[str, Dict[str, Any]],
        run: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        self.latched = latched
        self.run = run
        self.published = threading.Event()

    def socket(self, *_args: Any, **_kwargs: Any) -> "FakeSocket":
        return FakeSocket(self)


class FakeSocket:
    def __init__(self, bridge: FakeBridge) -> None:
        self.bridge = bridge
        self.timeout: Optional[float] = None
        self.topic: Optional[str] = None
        self.queue: List[Dict[str, Any]] = []
        self.run_loaded = False

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def send(self, raw: str) -> None:
        msg = json.loads(raw)
        if msg["op"] == "subscribe":
            self.topic = msg["topic"]
            if self.topic in self.bridge.latched:
                self.queue.append(self.bridge.latched[self.topic])
        elif msg["op"] == "publish":
            self.bridge.published.set()

    def recv(self) -> str:
        if not self.queue and not self.run_loaded:
            if not self.bridge.published.wait(self.timeout or 5.0):
                raise websocket.WebSocketTimeoutException("no message")
            self.queue.extend(self.bridge.run.get(self.topic, []))
            self.run_loaded = True
        if not self.queue:
            raise websocket.WebSocketTimeoutException("no message")
        return json.dumps({"op": "publish", "topic": self.topic, "msg": self.queue.pop(0)})

    def close(self) -> None:
        pass


@pytest.fixture
def ws_manager() -> Mock:
    ws = Mock()
    ws.url = "ws://fake"
    # The cloud's marker becomes its single point, so a test can tell clouds apart.
    ws._parse_pointcloud.side_effect = lambda msg: (
        np.array([[msg["marker"], 0.0, 0.0]]), None, "arm_camera_color_optical_frame"
    )
    ws.send_action_goal.return_value = {
        "transform": {"transform": {
            "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }}
    }
    return ws


async def _segment(bridge: FakeBridge, ws: Mock, cache: Dict[str, Any], monkeypatch) -> Dict[str, Any]:
    monkeypatch.setattr(websocket, "create_connection", bridge.socket)
    ws._ensure_connected.return_value = bridge.socket()
    mcp = FastMCP("test-segmentation")
    register_segmentation_tools(mcp, ws, cache)
    async with Client(mcp) as client:
        result = await client.call_tool("segment_objects", {"prompt": "cube", "timeout": 3.0})
        raw = result.content[0].text if result.content else result.data
        return json.loads(raw) if isinstance(raw, str) else raw


class TestLatchedStatus:
    async def test_old_success_does_not_hide_a_new_miss(self, ws_manager, monkeypatch) -> None:
        bridge = FakeBridge(
            latched={STATUS: {"data": "SUCCESS"}, CLOUD: _cloud(1, 1.0)},
            run={STATUS: [{"data": "SEGMENTING"}, {"data": "NO_OBJECTS_FOUND"}]},
        )
        cache: Dict[str, Any] = {"points": np.array([[9.0, 0.0, 0.0]]), "frame_id": "old"}

        result = await _segment(bridge, ws_manager, cache, monkeypatch)

        assert result["status"] == "NO_OBJECTS_FOUND"
        assert cache == {}

    async def test_old_miss_does_not_hide_a_new_success(self, ws_manager, monkeypatch) -> None:
        bridge = FakeBridge(
            latched={STATUS: {"data": "NO_OBJECTS_FOUND"}, CLOUD: _cloud(1, 1.0)},
            run={STATUS: [{"data": "SEGMENTING"}, {"data": "SUCCESS"}], CLOUD: [_cloud(2, 2.0)]},
        )
        cache: Dict[str, Any] = {}

        result = await _segment(bridge, ws_manager, cache, monkeypatch)

        assert result["status"] == "SUCCESS"
        assert cache["points"][0][0] == 2.0

    async def test_late_old_error_is_not_taken_for_this_run(self, ws_manager, monkeypatch) -> None:
        # A previous run's error arriving after the drain, before this run's messages.
        bridge = FakeBridge(
            latched={},
            run={
                STATUS: [
                    {"data": "ERROR: Server timeout"},
                    {"data": "SEGMENTING"},
                    {"data": "SUCCESS"},
                ],
                CLOUD: [_cloud(2, 2.0)],
            },
        )
        cache: Dict[str, Any] = {}

        result = await _segment(bridge, ws_manager, cache, monkeypatch)

        assert result["status"] == "SUCCESS"
        assert cache["points"][0][0] == 2.0

    async def test_node_error_without_segmenting_is_returned(self, ws_manager, monkeypatch) -> None:
        bridge = FakeBridge(
            latched={STATUS: {"data": "SUCCESS"}},
            run={STATUS: [{"data": "ERROR: No image"}]},
        )

        result = await _segment(bridge, ws_manager, {}, monkeypatch)

        assert result["status"] == "ERROR: No image"


class TestMissingCloud:
    async def test_success_without_new_cloud_is_an_error(self, ws_manager, monkeypatch) -> None:
        bridge = FakeBridge(
            latched={CLOUD: _cloud(1, 1.0)},
            run={STATUS: [{"data": "SEGMENTING"}, {"data": "SUCCESS"}]},
        )
        cache: Dict[str, Any] = {"points": np.array([[9.0, 0.0, 0.0]]), "frame_id": "old"}

        result = await _segment(bridge, ws_manager, cache, monkeypatch)

        assert "error" in result
        assert "status" not in result
        assert cache == {}
