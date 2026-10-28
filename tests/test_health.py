"""Startup health check: it must test the TF buffer server the tools use."""

from unittest.mock import Mock

import websocket

import perception_mcp.main as main


def test_health_checks_canonical_tf_server(monkeypatch, capsys):
    monkeypatch.setattr(websocket, "create_connection", lambda *a, **k: Mock())
    ws = Mock()
    ws.send_action_goal.return_value = {"transform": {"translation": {}, "rotation": {}}}
    monkeypatch.setattr(main, "ws_manager", ws)

    main._startup_health_check()

    assert ws.send_action_goal.call_args.kwargs["action_name"] == "/canonical_tf/tf2_buffer_server"
    assert "[health] /canonical_tf/tf2_buffer_server OK" in capsys.readouterr().err
