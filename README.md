# perception-mcp-server

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy-22314e)
![MCP](https://img.shields.io/badge/MCP-server-orange)
![Last commit](https://img.shields.io/github/last-commit/michaljohnson/perception_mcp_server)

An [MCP](https://modelcontextprotocol.io) server that exposes a mobile-manipulation robot's perception primitives — text-prompted segmentation, top-down grasp planning, top-down place planning, and raw camera access — as tools an LLM agent can call.



https://github.com/user-attachments/assets/a2e26385-8f25-4c78-b2b7-41480a1236a6


It is the perception layer in a larger stack where an LLM-driven agent uses MCP tools to drive a real or simulated robot through pick / place / navigate tasks. Motion planning (MoveIt), navigation (nav2), and direct ROS topic access are typically handled by separate MCP servers.

## Features

* **Text-prompted object segmentation** — produces a pixel mask and a 3D point cloud of the requested object from a free-form prompt.
* **Top-down grasp planning** — converts a segmented cloud into a base-frame grasp pose, with gripper-finger offset and shape-aware yaw alignment (PCA-driven).
* **Top-down place planning** — produces a drop pose above a surface or into a container using simple statistics on the segmented cloud, with mode-specific vertical clearance.
* **Dual-camera support** — separate segmentation pipelines for a body-mounted forward camera (navigation / verification) and a wrist-mounted arm camera (manipulation).
* **Raw camera access** — `look` returns JPEG images directly to a multimodal client, keeping pixel-level reasoning in the caller.
* **ROS 2 via rosbridge** — point cloud capture, TF lookups, and image streams all go through rosbridge, decoupling the MCP server from a native rclpy build.

## Quick start

```bash
git clone https://github.com/michaljohnson/perception_mcp_server.git
cd perception_mcp_server
uv sync                              # or: pip install -e .

cp .env.example .env                 # then edit ROSBRIDGE_IP, _PORT, SAM3_REMOTE_URL

uv run perception-mcp-server         # default transport: stdio
```

For HTTP transport:

```bash
uv run perception-mcp-server --transport streamable-http --host 0.0.0.0 --port 8003
```

### MCP client configuration

Drop one of these into your client's `mcp.json` (Claude Desktop, Cursor, VS Code, etc.).

**stdio (local):**

```json
{
  "perception mcp server": {
    "type": "stdio",
    "command": "uv",
    "args": [
      "run",
      "--directory",
      "/path/to/perception_mcp_server",
      "perception-mcp-server"
    ]
  }
}
```

**HTTP transport:**

```json
{
  "perception mcp server": {
    "type": "http",
    "url": "http://localhost:8003/mcp"
  }
}
```

## Tools

| Tool | Returns | When to use |
|---|---|---|
| `segment_objects` | Status + segmented point cloud (cached internally) | Always first — feeds the cache the other tools read. |
| `get_topdown_grasp_pose` | Top-down `grasp_pose` in robot base frame | After `segment_objects` on the **arm** camera. For picking. |
| `get_topdown_placing_pose` | Top-down `place_pose` in robot base frame | After `segment_objects`. For placing onto a surface or into a container. |
| `look` | Raw camera frame(s) as JPEG `Image` content | Area verification, gripper-state checks, or any "what does the robot see?" question. |

### Typical call sequence

Pick:

```
segment_objects(prompt="red cup", camera="arm")
  → status: SUCCESS, point cloud cached
get_topdown_grasp_pose(object_name="red cup")
  → grasp_pose at (x, y, z), top-down orientation
[hand off to a motion-planning MCP to move the arm]
```

Pick + place:

```
segment_objects(prompt="trash bin", camera="arm")
  → status: SUCCESS
get_topdown_placing_pose(object_name="trash bin", top_clearance_m=0.35)
  → place_pose above the bin
[hand off to motion planning to release the held object]
```

---

### `segment_objects(prompt, camera="arm", timeout=30.0)`

Sends `prompt` (free-form text, e.g. `"red cup"`, `"black scissors"`, `"tall rectangular bin"`) to a remote segmentation pipeline running as a ROS node. The node combines a vision-language detector with a segmentation model to produce a pixel mask plus a 3D point cloud of just the requested object.

The actual segmentation runs **outside this server**. The MCP tool publishes the prompt to a ROS topic, waits for a status reply, and captures the resulting point cloud over a dedicated websocket subscription.

**Cameras**

| `camera` | Use for |
|---|---|
| `"arm"` (default) | Wrist-mounted camera. Close-range manipulation. |
| `"front"` | Body-mounted forward camera. Navigation / approach verification. |

Each camera has its own segmentation node on disjoint topic prefixes:

| | Arm | Front |
|---|---|---|
| Trigger | `/segment_text` | `/front/segment_text` |
| Status | `/segmentation_status` | `/front/segmentation_status` |
| Mask | `/segmentation_mask` | `/front/segmentation_mask` |
| Point cloud | `/segmented_pointcloud` | `/front/segmented_pointcloud` |

**Cache.** On `SUCCESS` the tool caches the segmented cloud (points, colors, frame_id) together with the TF transform from the camera optical frame to the robot base frame, captured at the instant of segmentation. The TF snapshot pins the transform that was valid for those specific points, so a later `get_topdown_grasp_pose` call still computes the correct base-frame pose even if the arm has moved.

**Returns**

```json
{"status": "SUCCESS" | "NO_OBJECTS_FOUND" | "<other>",
 "prompt": "<prompt>", "camera": "arm" | "front",
 "outputs": {"mask_topic": "...", "pointcloud_topic": "..."},
 "description": "..."}
```

---

### `get_topdown_grasp_pose(object_name, pointcloud_topic="/segmented_pointcloud", timeout=10.0)`

Computes a top-down grasp pose from the cached point cloud (or from a fresh subscription on `pointcloud_topic` as a fallback). Uses centroid-based positioning, a configurable gripper-finger vertical offset (`GRIPPER_FINGER_OFFSET_M` in `utils/transforms.py`, default 0.14 m for Robotiq 2F-140), and a 2D PCA on the base-frame `(x, y)` projection for shape-aware yaw alignment.

When the segmented object is elongated (aspect ratio ≥ 1.2), the gripper yaw is rotated so the fingers close across the short axis. Otherwise the orientation falls back to strict top-down `(x=1, y=0, z=0, w=0)`.

**Returns (success)**

```json
{"object_name": "...",
 "centroid_base_frame":   {"x":..,"y":..,"z":..},
 "grasp_pose": {"frame_id":"base_footprint",
                "position":{"x":..,"y":..,"z":..},
                "orientation":{"x":..,"y":..,"z":..,"w":..}},
 "bounding_box": {"min":{...}, "max":{...}, "size":{...}},
 "num_points": N,
 "principal_axis_aspect_ratio": ..., "oriented": true|false}
```

If TF fails the response includes `centroid_camera_frame` only and a `warning` field — callers should treat this as a failure.

**Limitations**

- Approach direction is fixed (-z in base frame). Only the yaw is shape-aware. Side or angled grasps need a different primitive.
- Centroid-based position works for compact, roughly symmetric objects. Long or irregular objects need a more sophisticated grasp point.
- Gripper finger-axis convention assumes Robotiq 2F-140 mounted such that fingers close along the gripper-tool X axis. For tool-Y mounts, remove the `+ math.pi / 2` term in `grasping.py`.

**Prerequisites.** Call `segment_objects(camera="arm", ...)` first. Front-camera grasps are unreliable because the front camera's geometry is not optimized for close-range manipulation.

---

### `get_topdown_placing_pose(object_name, top_clearance_m=0.20, ...)`

Computes a top-down drop pose using simple statistics on the cached point cloud: horizontal centroid for `x, y`, 95th percentile of `z` for the surface top (avoids being lifted into the air by stray noisy points), plus `top_clearance_m` for vertical safety margin.

The same algorithm handles surfaces and containers — only the clearance differs:

| Target type | Suggested `top_clearance_m` |
|---|---|
| Surface (table, counter, shelf, desk) | `0.20` |
| Container (bin, basket, bowl, box) | `0.35` |

**Modes**

- `use_cached=True` (default) — use the cloud cached by `segment_objects`.
- `use_cached=False` — read a raw depth topic, optionally cropped to a base-frame xy region (`crop_center_x`, `crop_center_y`, `crop_radius_m`). Useful when segmentation fails on a given view and a coarse target xy is available from a previous step.

**Returns (success)**

```json
{"object_name": "...",
 "surface_height_m": <top_z>,
 "surface_centroid": {"x":cx,"y":cy,"z":top_z},
 "place_pose": {"frame_id":"base_footprint",
                "position":{"x":cx,"y":cy,"z":drop_z},
                "orientation":{"x":1,"y":0,"z":0,"w":0}},
 "top_clearance_m": <clearance>,
 "num_points_used": N}
```

**Limitations**

- The xy mean is biased toward whichever side of the target the camera sees more of. A bin viewed horizontally has its near rim oversampled, pulling the centroid forward. Mitigation: drive close enough that the bias is small, or position the arm directly above the coarse target and re-segment from above.
- Raw-depth mode (`use_cached=False`) bypasses segmentation, so the crop must be tight.

---

### `look(camera="front")`

Returns the current frame from the requested camera as a JPEG `Image` content block (the FastMCP standard image type). A multimodal LLM client receives the bytes directly and can reason on the pixels with its own vision capability.

| `camera` | Returns |
|---|---|
| `"front"` (default) | Single `Image` of the body-mounted forward camera. |
| `"arm"` | Single `Image` of the wrist-mounted camera. |
| `"both"` | `list[Image]` of `[front, arm]`. |

**When to use**

- Area / room verification after navigation (`camera="both"` for the combined view).
- Gripper-state checks after pick / place (`camera="front"` shows the gripper in the upper part of the frame when the arm is in `look_forward`).
- General "what does the robot see right now?" debugging.

Not appropriate for grasp / place planning — use `segment_objects` plus `get_topdown_grasp_pose` / `get_topdown_placing_pose` for pixel-accurate masks and 3D points.

---

## Configuration

`load_dotenv` reads `.env` at the server root if present.

| Var | Default | Purpose |
|---|---|---|
| `ROSBRIDGE_IP` | `127.0.0.1` | rosbridge host. |
| `ROSBRIDGE_PORT` | `9090` | rosbridge port. |
| `SAM3_REMOTE_URL` | — | Optional health check at startup. If set, a missing segmentation backend fails loud at server start rather than silently inside `segment_objects`. |

## Prerequisites at runtime

- `rosbridge_websocket` reachable on `ROSBRIDGE_IP:ROSBRIDGE_PORT`.
- `tf2_buffer_server` action server running, so TF lookups succeed (LookupTransform is an action in ROS 2 Jazzy and later, not a service).
- For `segment_objects`: at least one segmentation ROS node running (arm camera, front camera, or both), backed by a reachable segmentation server.
- For `look`: at least one of the two RGB camera topics publishing as `sensor_msgs/CompressedImage` (defaults: `/front_rgbd_camera/color/image_raw/compressed`, `/arm_camera/color/image_raw/compressed`).

## Layout

```
perception-mcp-server/
├── server.py                       # entry point; CLI args + transport
├── pyproject.toml
├── perception_mcp/
│   ├── main.py                     # tool registration + health checks
│   ├── tools/
│   │   ├── segmentation.py         # segment_objects
│   │   ├── grasping.py             # get_topdown_grasp_pose
│   │   ├── placing.py              # get_topdown_placing_pose
│   │   └── detection.py            # look
│   └── utils/
│       ├── websocket.py            # rosbridge / TF2 / topic I/O
│       └── transforms.py           # TOP_DOWN_ORIENTATION + TF helpers
```

## Limitations

- **Top-down only.** Grasp and place poses are always strictly top-down. Side or angled approaches are not supported.
- **Single internal cache.** `segment_objects` overwrites one cache slot. No per-camera cache, no history, not thread-safe under parallel calls — typical agent flows call perception sequentially.
- **Coupled to rosbridge.** All ROS I/O goes through rosbridge. A native rclpy build is possible but not implemented.

## License

See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome — please open an issue or PR on GitHub.
