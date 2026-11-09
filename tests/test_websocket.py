"""Tests for ``WebSocketManager`` against fake rosbridge sockets."""

import json
from typing import Any, Dict, Optional

import pytest
import websocket

from perception_mcp.utils.websocket import WebSocketManager

VALUES = {"transform": {"transform": {"translation": {"x": 0.0, "y": 0.0, "z": 0.0}}}}


class FakeSocket:
    """Answers the goal it was sent with one rosbridge action_result."""

    def __init__(self, result: bool, status: Optional[int]) -> None:
        self.result = result
        self.status = status
        self.goal_id = None

    def send(self, raw: str) -> None:
        self.goal_id = json.loads(raw)["id"]

    def settimeout(self, _timeout: float) -> None:
        pass

    def recv(self) -> str:
        msg: Dict[str, Any] = {
            "op": "action_result",
            "id": self.goal_id,
            "values": VALUES,
            "result": self.result,
        }
        if self.status is not None:
            msg["status"] = self.status
        return json.dumps(msg)


def _send(result: bool, status: Optional[int]) -> dict:
    manager = WebSocketManager("localhost", 9090)
    manager._ensure_connected = lambda: FakeSocket(result, status)
    return manager.send_action_goal(
        "/canonical_tf/tf2_buffer_server", "tf2_msgs/action/LookupTransform", {}
    )


class TestSendActionGoal:
    def test_succeeded_returns_values(self) -> None:
        assert _send(result=True, status=4) == VALUES

    def test_aborted_raises(self) -> None:
        # rosbridge reports a finished but aborted goal with result=True.
        with pytest.raises(RuntimeError, match="status 6"):
            _send(result=True, status=6)

    def test_canceled_raises(self) -> None:
        with pytest.raises(RuntimeError, match="status 5"):
            _send(result=True, status=5)

    def test_without_status_returns_values(self) -> None:
        # Older rosbridge versions send no status.
        assert _send(result=True, status=None) == VALUES

    def test_failed_call_raises(self) -> None:
        with pytest.raises(RuntimeError, match="failed"):
            _send(result=False, status=None)


class StreamingBridge:
    """Each subscribe makes rosbridge send three frames, numbered in send order."""

    def __init__(self) -> None:
        self.sent = 0
        self.sockets = []

    def socket(self, *_args, **_kwargs) -> "StreamingSocket":
        sock = StreamingSocket(self)
        self.sockets.append(sock)
        return sock


class StreamingSocket:
    def __init__(self, bridge: StreamingBridge) -> None:
        self.bridge = bridge
        self.queue = []
        self.connected = True

    def send(self, raw: str) -> None:
        msg = json.loads(raw)
        if msg["op"] == "subscribe":
            for _ in range(3):
                self.bridge.sent += 1
                self.queue.append(json.dumps({"topic": msg["topic"], "msg": {"seq": self.bridge.sent}}))

    def settimeout(self, _timeout: float) -> None:
        pass

    def recv(self) -> str:
        if not self.queue:
            raise websocket.WebSocketTimeoutException("no data")
        return self.queue.pop(0)

    def close(self) -> None:
        self.connected = False


class TestSubscribeOnce:
    def test_second_call_gets_a_frame_sent_after_its_subscribe(self, monkeypatch) -> None:
        bridge = StreamingBridge()
        monkeypatch.setattr(websocket, "create_connection", bridge.socket)
        manager = WebSocketManager("localhost", 9090)
        assert manager.subscribe_once("/img")["seq"] == 1
        # Frames 2 and 3 were sent after the first call returned; they must not come back.
        assert manager.subscribe_once("/img")["seq"] == 4

    def test_connection_closed_after_message(self, monkeypatch) -> None:
        bridge = StreamingBridge()
        monkeypatch.setattr(websocket, "create_connection", bridge.socket)
        WebSocketManager("localhost", 9090).subscribe_once("/img")
        assert not bridge.sockets[0].connected

    def test_connection_closed_after_timeout(self, monkeypatch) -> None:
        silent = StreamingSocket(StreamingBridge())
        silent.send = lambda raw: None  # subscribed, but nothing ever arrives
        monkeypatch.setattr(websocket, "create_connection", lambda *a, **k: silent)
        with pytest.raises(TimeoutError):
            WebSocketManager("localhost", 9090).subscribe_once("/img", timeout=0.1)
        assert not silent.connected
