"""Example MCP client for perception-mcp-server.

Connects to a running perception-mcp-server, runs a complete pick-style
sequence on the arm camera, prints the results, and exits.

Prerequisites:
    - perception-mcp-server is running on http://localhost:8003 with HTTP
      transport. Start it with:
          python server.py --transport streamable-http --port 8003
    - rosbridge_websocket is reachable on ROSBRIDGE_IP:ROSBRIDGE_PORT.
    - A segmentation ROS node is running on the arm camera, backed by a
      reachable Grounding-DINO + SAM HTTP service.
    - At least one object matching `--prompt` is in the arm camera's view.

Usage:
    python examples/test_client.py --prompt "red cup"
    python examples/test_client.py --prompt "scissors" --server-url http://192.168.1.10:8003/mcp

The output prints the SAM3 status, the centroid in camera and base
frames, the grasp pose ready for MoveIt, and the placement pose for a
typical surface drop. No motion is executed — this client only exercises
the perception side of a pick sequence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

try:
    from fastmcp import Client
except ImportError:
    print("fastmcp not installed. Install with: pip install fastmcp", file=sys.stderr)
    sys.exit(1)


async def run(server_url: str, prompt: str, camera: str) -> int:
    """Run a segmentation + grasp + place query against the server."""
    print(f"Connecting to {server_url} ...")
    async with Client(server_url) as client:
        # Quick sanity: list the registered tools.
        tools = await client.list_tools()
        tool_names = sorted(t.name for t in tools)
        print(f"Server exposes {len(tool_names)} tools: {', '.join(tool_names)}")

        # 1. Segment the object.
        print(f"\n[1] segment_objects(prompt={prompt!r}, camera={camera!r}) ...")
        seg = await client.call_tool(
            "segment_objects",
            {"prompt": prompt, "camera": camera, "timeout": 30},
        )
        seg_payload = _decode(seg)
        print(json.dumps(seg_payload, indent=2))
        if seg_payload.get("status") != "SUCCESS":
            print(
                "\nSegmentation did not return SUCCESS — cannot continue with "
                "grasp / place. Check the segmentation node and backend; see "
                "docs/TROUBLESHOOTING.md."
            )
            return 1

        # 2. Compute a top-down grasp pose for the segmented object.
        print(f"\n[2] get_topdown_grasp_pose(object_name={prompt!r}) ...")
        grasp = await client.call_tool(
            "get_topdown_grasp_pose", {"object_name": prompt}
        )
        grasp_payload = _decode(grasp)
        print(json.dumps(grasp_payload, indent=2))

        # 3. Compute a surface-mode place pose using a plausible
        #    object height (e.g. 12cm for a coke can) and a 5cm air gap.
        print(
            f"\n[3] get_topdown_placing_pose(object_name={prompt!r}, "
            f"object_height_m=0.12, top_clearance_m=0.05) ..."
        )
        place = await client.call_tool(
            "get_topdown_placing_pose",
            {
                "object_name": prompt,
                "object_height_m": 0.12,
                "top_clearance_m": 0.05,
            },
        )
        place_payload = _decode(place)
        print(json.dumps(place_payload, indent=2))

    print("\nDone. No motion executed; pass grasp_pose / place_pose to a "
          "motion-planning MCP (e.g. moveit) to actually move the arm.")
    return 0


def _decode(result) -> dict:
    """Extract the tool's JSON payload from a FastMCP result object."""
    if not getattr(result, "content", None):
        return {}
    block = result.content[0]
    text = getattr(block, "text", None) or block
    if isinstance(text, str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
    return {"raw": str(text)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--server-url",
        default="http://localhost:8003/mcp",
        help="MCP server URL (default: http://localhost:8003/mcp).",
    )
    ap.add_argument(
        "--prompt",
        default="red cup",
        help="Object prompt to segment (default: 'red cup').",
    )
    ap.add_argument(
        "--camera",
        default="arm",
        choices=["arm", "front"],
        help="Which camera to segment with (default: arm).",
    )
    args = ap.parse_args()

    sys.exit(asyncio.run(run(args.server_url, args.prompt, args.camera)))


if __name__ == "__main__":
    main()
