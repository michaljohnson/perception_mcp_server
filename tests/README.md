# tests

Unit tests for perception-mcp-server. Pure-Python — no rosbridge,
no live ROS stack, no segmentation backend required. The
WebSocketManager and the segmentation cache are mocked.

## Layout

| File | Coverage |
|---|---|
| `conftest.py` | Shared fixtures: identity TF dicts, sample point clouds (elongated and round), sample segmentation cache, mocked WebSocketManager. |
| `test_transforms.py` | Pure-math helpers in `perception_mcp/utils/transforms.py`: quaternion-to-rotation matrix, point and array transforms, PCA principal-axis extraction, grasp-yaw wrapping, quaternion multiplication, composed top-down quaternion, and `_tf_lookup` with a mocked WebSocketManager. |
| `test_detection.py` | `look(camera=...)` tool: front, arm, both, default, invalid value, websocket error propagation. |
| `test_grasping.py` | `get_topdown_grasp_pose` tool: cached cloud happy path, gripper finger offset, cached-TF preference, empty cache fallback to topic, timeout, empty point cloud. |
| `test_placing.py` | `get_topdown_placing_pose` tool: container mode, surface mode (finger + object-height + clearance), x_bias forward-offset, raw-depth fallback, timeout. |

## Running

```bash
uv sync --all-extras
uv run pytest tests/
```

If the run errors with `ModuleNotFoundError: No module named 'lark'`
(or similar from `/opt/ros/<distro>/...`), the surrounding shell has
sourced a ROS environment and leaked `PYTHONPATH` into the test
subprocess. Clear it for the command:

```bash
PYTHONPATH= uv run pytest tests/
```

Cleanest long-term fix: run perception tests from a terminal that has
NOT sourced ROS. The perception tests are pure Python and do not need
the ROS environment at all.

## Conventions

- **Neutral voice.** Test docstrings describe behavior, not history.
  No first-person pronouns, no calendar dates, no incident references.
- **Deterministic randomness.** Any `numpy.random` use seeds via
  `default_rng(seed=...)` so test failures are reproducible.
- **Mock at the boundary.** The WebSocketManager is the only ROS
  contact point — mock it at the call site. Don't mock internal
  helpers (`_quat_to_rotation_matrix`, `_principal_axis_angle_xy`,
  etc.); they are pure and should run as written.
- **Tool tests use the FastMCP `Client`.** Each tool test registers
  the tool against a fresh `FastMCP` instance, calls it via a
  `Client`, and inspects either the JSON result or the mock call
  arguments. This mirrors how the server is actually exercised at
  runtime.

## Extending

When adding a new tool to `perception_mcp/tools/`:

1. Add a fixture in `conftest.py` if it needs new sample data
   (point clouds, TF dicts, cache state).
2. Create `tests/test_<tool>.py` mirroring `test_detection.py` or
   `test_grasping.py` — register the tool against a `FastMCP`, call
   it via `Client`, assert on outputs and mock invocations.
3. Cover the happy path, parameter validation, and at least one
   error-propagation case (timeout or upstream exception).

When adding a pure helper to `perception_mcp/utils/transforms.py`:

1. Add the test class in `test_transforms.py` next to the related
   helpers.
2. Include the degenerate / numerically-zero edge case if the
   helper has one (PCA on collinear points, identity quaternion,
   etc.).

## Not yet covered

- `segment_objects` orchestration in `tools/segmentation.py` —
  requires non-trivial rosbridge mocking (publish, status poll,
  pointcloud subscribe, cache write). Add when the segmentation
  cache becomes load-bearing for a new feature.
- The live-TF fallback path inside `get_topdown_grasp_pose` and
  `get_topdown_placing_pose` (when the cache has no `tf_translation`
  but the call is still expected to succeed by calling `_tf_lookup`
  through the mocked WebSocketManager). The cached-TF path covers
  the dominant case; the live path is exercised indirectly by the
  empty-cache fallback tests.
