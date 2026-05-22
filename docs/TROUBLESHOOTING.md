# Troubleshooting

Common failure modes and how to diagnose them.

## `segment_objects` returns `NO_OBJECTS_FOUND` for everything

The most likely cause: the segmentation ROS node is not running, or
the external Grounding-DINO + SAM backend is unreachable.

**Check the segmentation node is running:**

```bash
ros2 topic list | grep segment
```

Expected output for arm + front nodes:

```
/segment_text
/segmentation_status
/segmentation_mask
/segmented_pointcloud
/front/segment_text
/front/segmentation_status
/front/segmentation_mask
/front/segmented_pointcloud
```

If only one prefix is present, only that camera will work; the other
returns `NO_OBJECTS_FOUND` or `UNKNOWN`.

**Check the backend HTTP service is reachable** from the host running
the segmentation node:

```bash
curl -I "$SAM3_REMOTE_URL"
```

A timeout, 502, or connection refused indicates the backend is down.
Restart the backend or point the segmentation node at a different URL.

## `segment_objects` returns `UNKNOWN`

The MCP server published the prompt but did not receive a status reply
within the timeout. Causes in order of likelihood:

1. Segmentation node is not running on the queried camera.
2. The status topic name does not match what the server expects.
3. The backend is responding slowly, exceeding the request timeout.

Raise the `timeout` argument on the tool call (default 30 s) or
investigate the segmentation node logs.

## `get_topdown_grasp_pose` returns `centroid_camera_frame` only and a `warning` field

The TF lookup from camera frame to `base_footprint` failed. Causes:

1. The canonical tf2 buffer-server action is not running. Start it
   under namespace `/canonical_tf` with `use_sim_time:=true` so
   simulated clocks line up.
2. The camera frame ID in the cached point cloud is not in the TF
   tree (typo in camera config, or the camera driver is not
   publishing a frame).
3. The cached TF snapshot inside `segmentation_cache` is malformed.

Check the running buffer-servers:

```bash
ros2 action list | grep buffer_server
```

There should be exactly one `/canonical_tf/tf2_buffer_server`. If
multiple appear (or if the canonical-namespace one is missing), the
TF lookup will race against unrelated default-named servers — see
**Architecture** for the rationale.

## `look(camera="arm")` times out

The arm camera RGB topic is not publishing as
`sensor_msgs/CompressedImage`. Verify:

```bash
ros2 topic info /arm_camera/color/image_raw/compressed
ros2 topic hz /arm_camera/color/image_raw/compressed
```

If the topic does not exist or is silent, the camera driver is not
running. If the type is `sensor_msgs/Image` (raw), set the camera
driver to publish `compressed` or add an image republisher.

## Rosbridge connection errors

`websocket._exceptions.WebSocketConnectionClosedException` or
`Connection refused` on startup:

```bash
echo "$ROSBRIDGE_IP"   # expected: 127.0.0.1 by default
echo "$ROSBRIDGE_PORT" # expected: 9090

# Is rosbridge_server actually running?
ros2 node list | grep rosbridge
```

If no `rosbridge` node is listed, launch one:

```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

If multiple are listed, you have stale instances from previous runs.
Kill them with `pkill -9 -f rosbridge` before restarting.

## Multiple `tf_buffer` instances in `ros2 node list`

After a few restart cycles the system can accumulate stale
`tf2_buffer_server` and `rosapi` processes that survived previous
launches. They appear in `ros2 node list` as duplicates of the same
name and cause non-deterministic action routing.

Cleanup:

```bash
pkill -9 -f rosbridge
pkill -9 -f rosapi
pkill -9 -f buffer_server
ros2 daemon stop && sleep 2 && ros2 daemon start
```

Then relaunch your stack from a clean state.

## DDS discovery ghosts

Even after killing the processes, `ros2 node list` may show dead nodes
for up to ~60 seconds because DDS's discovery cache holds them until
the heartbeat timeout. If the cache is bothering you sooner:

```bash
ros2 daemon stop
ros2 daemon start
```

That clears the local discovery state. Genuinely-running nodes will
reappear within a few seconds.

## Tests fail with `ModuleNotFoundError: No module named 'lark'`

The pytest run is picking up a broken `launch_testing` plugin from the
system ROS install via a leaked `PYTHONPATH`. Clear it for the
command:

```bash
PYTHONPATH= uv run pytest tests/
```

Or open a new terminal that has not sourced any ROS distro. The
perception tests are pure-Python and do not need a ROS environment.
