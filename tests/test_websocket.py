"""Tests for ``WebSocketManager.send_action_goal`` against a fake rosbridge socket."""

import json
from typing import Any, Dict, Optional

import pytest

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
