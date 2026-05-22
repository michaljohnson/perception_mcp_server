# Architecture

How perception-mcp-server is structured internally, what each module does,
and how data flows from a user prompt to a base-frame pose.

## High-level shape

The server is a thin MCP wrapper around two external systems:

```
┌───────────────┐         ┌───────────────────┐
│  LLM agent    │         │  rosbridge_server │
│  (any MCP     │ MCP ───▶│  (websocket on    │
│  client)      │         │  port 9090)       │
└───────────────┘         └─────────┬─────────┘
        ▲                           │
        │                           │ ROS 2 topics / actions
        │                           ▼
        │                 ┌───────────────────┐
        │                 │  ROS 2 stack:     │
        │                 │  - segmentation   │
        │                 │    nodes          │
        │                 │  - camera         │
        │                 │    drivers        │
        │                 │  - TF2            │
        │                 └─────────┬─────────┘
        │                           │
        │                           │ HTTP
        │                           ▼
        │                 ┌───────────────────┐
        └──── tool ◀──────│  Grounding-DINO + │
              results     │  SAM (HTTP API)   │
                          └───────────────────┘
```

The server does no learning of its own. The heavy lifting (vision-language
detection, segmentation) runs in an external HTTP service that the
segmentation ROS node talks to.

## Modules

```
src/perception_mcp/
├── main.py                       Server entry: parses CLI args, registers
│                                 tools, calls FastMCP.run().
├── tools/
│   ├── segmentation.py           segment_objects() — publish a prompt to
│   │                             the segmentation node, wait for the
│   │                             status reply, capture the resulting
│   │                             point cloud, cache it.
│   ├── grasping.py               get_topdown_grasp_pose() — read the
│   │                             cached cloud, compute centroid + 2D
│   │                             PCA, apply gripper finger offset,
│   │                             return a base-frame pose.
│   ├── placing.py                get_topdown_placing_pose() — read the
│   │                             cached cloud, mean(xy) + p95(z),
│   │                             apply mode-specific clearance, return
│   │                             a base-frame pose.
│   └── detection.py              look() — fetch raw JPEG frames from
│                                 the camera topics and return them as
│                                 FastMCP Image content blocks.
└── utils/
    ├── websocket.py              WebSocketManager — rosbridge client.
    │                             Wraps publish, subscribe-once, call
    │                             service, send action goal.
    └── transforms.py             Pure-math helpers: quaternion ops, 2D
                                  PCA principal-axis, finger-offset
                                  constant, _tf_lookup against the
                                  canonical buffer-server action.
```

## Data flow: a typical pick sequence

```
1. Agent calls
       segment_objects(prompt="red cup", camera="arm")
   │
   ▼
2. WebSocketManager publishes the prompt to ROS topic
       /segment_text (arm camera)
   │
   ▼
3. Segmentation ROS node reads the prompt, queries the external
   Grounding-DINO + SAM HTTP backend, produces a mask + 3D point
   cloud.
   │
   ▼
4. The server subscribes to /segmentation_status and
   /segmented_pointcloud, captures the cloud.
   │
   ▼
5. The cloud + camera→base TF snapshot at this instant + diagnostics
   are written into the in-memory segmentation_cache.
   │
   ▼
6. Agent calls
       get_topdown_grasp_pose(object_name="red cup")
   │
   ▼
7. The tool reads the cache, computes centroid + 2D PCA, transforms
   to base_footprint using the cached TF, applies the gripper
   finger offset, returns the grasp pose.
   │
   ▼
8. Agent passes the grasp pose to a motion-planning MCP (MoveIt).
```

## Why the cache holds a TF snapshot

The point cloud is in the camera optical frame at capture time. By the
time `get_topdown_grasp_pose` runs, the arm (and therefore the camera)
may have moved. A live TF lookup at compute time would apply the
*current* transform to a *past* cloud, producing a geometrically wrong
base-frame pose. The cache pins the transform that was valid for those
specific points so the conversion stays consistent.

## Why the TF lookup uses the canonical buffer-server

In ROS 2 Jazzy, `LookupTransform` is an action, not a service. Multiple
processes (MoveIt, RViz, segmentation nodes, etc.) each spawn their own
`tf2_ros.Buffer.create_server()` and advertise the default action name.
ROS 2 discovery routes a goal request to whichever one binds first,
producing intermittent wrong transforms (camera-frame depth leaking
through as base-frame z, etc.). The server addresses this by calling a
namespaced canonical action at `/canonical_tf/tf2_buffer_server`,
launched separately so it never competes with the default-named servers.

## Concurrency

The shared `segmentation_cache` dict and the `WebSocketManager`
connection are **not thread-safe under parallel calls**. Sequential
agent flows are safe because tool calls execute one at a time at the
FastMCP boundary. Adding concurrent calls (e.g. parallel arm + front
segmentation in the same request) would require a `threading.Lock`
around cache writes and either per-camera cache partitioning or a
websocket-per-camera wrap.

## Where the model lives

Inside `perception-mcp-server`: no learned model, no GPU, no
PyTorch dependency. The segmentation pipeline (Grounding-DINO + SAM)
runs as a separate HTTP service that the ROS segmentation node
queries. This keeps the MCP server CPU-only, fast to start, and easy
to deploy on a robot's onboard compute.
