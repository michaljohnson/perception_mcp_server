"""Grasp planning tools.

Computes grasp targets from segmented point clouds, transforms them
to the robot base frame via TF2, and returns ready-to-use grasp poses
for MoveIt execution.
"""

import math

import numpy as np
from fastmcp import FastMCP

from perception_mcp.utils.transforms import (
    GRIPPER_FINGER_OFFSET_M,
    TOP_DOWN_ORIENTATION,
    _PCA_ASPECT_RATIO_MIN,
    _grasp_xy,
    _MAX_RANGE_M,
    _MIN_KEPT_POINTS,
    _grasp_z,
    _oriented_topdown_quaternion,
    _plausible,
    _principal_axis_angle_xy,
    _shortest_grasp_yaw,
    _tf_lookup,
    _transform_point,
    _transform_points,
)
from perception_mcp.utils.websocket import WebSocketManager


def _camera_stats(points) -> tuple[dict, dict]:
    """Centroid and bounding box of the cloud in the camera frame."""
    centroid = points.mean(axis=0)
    pt_min = points.min(axis=0)
    pt_max = points.max(axis=0)
    size = pt_max - pt_min

    def r(v) -> dict:
        return {"x": round(float(v[0]), 4), "y": round(float(v[1]), 4), "z": round(float(v[2]), 4)}

    return r(centroid), {"min": r(pt_min), "max": r(pt_max), "size": r(size)}


def register_grasping_tools(
    mcp: FastMCP,
    ws_manager: WebSocketManager,
    segmentation_cache: dict = None,
) -> None:
    """Register grasp planning tools."""
    if segmentation_cache is None:
        segmentation_cache = {}

    @mcp.tool(
        description=(
            "Compute a grasp pose from the segmented point cloud.\n\n"
            "Uses the cached point cloud from the last segment_objects() call.\n"
            "Computes the centroid, transforms it from camera frame to base_footprint\n"
            "via TF2, and returns a top-down grasp pose ready for MoveIt.\n"
            "The grasp x/y is the object's near face (seen from the camera) plus\n"
            "half its width, not the centroid: the camera sees only the front of\n"
            "an object. grasp_xy_method says which rule applied.\n\n"
            "The grasp pose accounts for the Robotiq 140 gripper finger length:\n"
            f"the wrist is {GRIPPER_FINGER_OFFSET_M} m above the pad middle. The gripper\n"
            "goes as deep as allowed: the fingertips 1 cm above the object's lowest\n"
            "seen points (never into the floor or the surface under the object), but\n"
            "the pads no more than 5 cm below the object's top\n"
            "(tall objects would hit the palm). pre_grasp_pose is the same pose\n"
            "with the fingertips 10 cm above the object's top: go there first, then\n"
            "descend to grasp_pose. grasp_z_method says which height rule applied.\n"
            "Not handled: long objects pointing at the camera (x lands too close).\n"
            "top_z is the object's top for setting something down on it (99th\n"
            "percentile of the heights; bbox z_max is the single highest point).\n\n"
            "Orientation is shape-aware: a 2D PCA on the base-frame xy\n"
            "projection of the point cloud gives the object's principal\n"
            "axis, and the gripper yaw is set so the fingers close ACROSS\n"
            "the short axis. For approximately round / square top-down\n"
            f"footprints (aspect ratio < {_PCA_ASPECT_RATIO_MIN}) the default top-down\n"
            "orientation [1, 0, 0, 0] is returned unchanged. The response\n"
            "exposes principal_axis_angle_rad, principal_axis_angle_deg,\n"
            "principal_axis_aspect_ratio, and a boolean oriented flag so\n"
            "callers can inspect the geometry.\n\n"
            "PREREQUISITE: Call segment_objects() first to generate the point cloud.\n\n"
            "The returned grasp_pose can be passed directly to MoveIt's\n"
            "plan_to_pose or plan_and_execute tool.\n\n"
            "Example usage:\n"
            "- First: segment_objects(prompt='scissors')\n"
            "- Then: get_topdown_grasp_pose(object_name='scissors')\n"
            "- Then: plan_and_execute with the returned grasp_pose\n"
        ),
    )
    def get_topdown_grasp_pose(
        object_name: str,
        pointcloud_topic: str = "/segmented_pointcloud",
        timeout: float = 10.0,
    ) -> dict:
        """Compute grasp pose from a segmented point cloud.

        Args:
            object_name: Name of the segmented object (for labeling).
            pointcloud_topic: Topic to read the point cloud from.
            timeout: Max seconds to wait for point cloud message.

        Returns:
            dict with centroid (camera + base frame), grasp_pose for MoveIt,
            and bounding box dimensions.
        """
        # Get pointcloud from cache or topic
        if segmentation_cache.get("points") is not None:
            points = segmentation_cache["points"]
            frame_id = segmentation_cache["frame_id"]
        elif pointcloud_topic.endswith("segmented_pointcloud"):
            # Latched: reading it would return an earlier segmentation's cloud.
            return {
                "error": (
                    "No segmentation result cached: the last segment_objects() "
                    "call failed or none ran. Call segment_objects() first."
                )
            }
        else:
            try:
                points, _, frame_id = ws_manager.get_pointcloud(
                    pointcloud_topic, timeout=timeout
                )
            except TimeoutError:
                return {
                    "error": (
                        f"No cached point cloud and timeout ({timeout}s) waiting "
                        f"on {pointcloud_topic}. Call segment_objects() first."
                    )
                }
            except Exception as e:
                return {"error": f"Failed to read point cloud: {str(e)}"}

        if len(points) == 0:
            return {
                "object_name": object_name,
                "error": "Point cloud is empty — segmentation may have failed.",
            }

        centroid_camera, bounding_box = _camera_stats(points)

        # TF lookup: camera frame → base_footprint.
        # Prefer the TF snapshot captured at segmentation time (so the
        # cached pointcloud is transformed with the TF that was valid when
        # it was captured). Fall back to a fresh lookup if unavailable.
        try:
            cached_t = segmentation_cache.get("tf_translation")
            cached_r = segmentation_cache.get("tf_rotation")
            if cached_t and cached_r and all(
                k in cached_t for k in ("x", "y", "z")
            ) and all(k in cached_r for k in ("x", "y", "z", "w")):
                translation, rotation = cached_t, cached_r
            else:
                translation, rotation = _tf_lookup(
                    ws_manager, source_frame=frame_id
                )
            points_base = _transform_points(points, translation, rotation)
            keep = _plausible(points, points_base)
            if keep.sum() < _MIN_KEPT_POINTS:
                return {
                    "object_name": object_name,
                    "error": (
                        f"only {int(keep.sum())} of {len(points)} points are plausible "
                        f"(rest below the floor or > {_MAX_RANGE_M} m away): bad depth on this object"
                    ),
                }
            dropped = int(len(points) - keep.sum())
            points, points_base = points[keep], points_base[keep]
            centroid_camera, bounding_box = _camera_stats(points)
            centroid_base = _transform_point(centroid_camera, translation, rotation)

            # 2D PCA on the base-frame xy projection of the point cloud
            # to align the gripper fingers with the object's short axis.
            # Round / square footprints (aspect ratio below the threshold)
            # fall back to the default top-down orientation.
            #
            # Robotiq 2F-140 finger axis: the
            # fingers close along the gripper-tool X axis, not tool Y.
            # After the top-down flip (q_topdown = 180 deg about base-X)
            # the finger axis maps to base +X. To rotate the fingers
            # ACROSS the object's short axis, the gripper yaw therefore
            # equals the SHORT-axis angle = angle_long + pi/2.

            # Base-frame bounding box. Approach needs this for wide-surface
            # targets where the centroid sits inside the object volume and
            # would put the standoff point past the near edge (e.g. coffee
            # table seen from the living-room entry). Camera-frame bbox
            # above is kept for back-compat but x/y/z axes there are
            # whichever the sensor publishes; this base-frame bbox is in
            # the conventional robot frame (x forward, y left, z up).
            # Computed in an inner try so that a bbox-only failure cannot
            # take down the main return path (the main except resets
            # centroid_base_frame to None which is a hard handoff break).
            bbox_base = {
                "x_min": round(float(points_base[:, 0].min()), 4),
                "x_max": round(float(points_base[:, 0].max()), 4),
                "y_min": round(float(points_base[:, 1].min()), 4),
                "y_max": round(float(points_base[:, 1].max()), 4),
                "z_min": round(float(points_base[:, 2].min()), 4),
                "z_max": round(float(points_base[:, 2].max()), 4),
            }
            # Top height for setting something down on the object: z_max is a single
            # point, which depth noise can put cm high; the 99th percentile (as in _grasp_z) is not.
            top_z = round(float(np.percentile(points_base[:, 2], 99)), 4)

            angle_long, aspect_ratio = _principal_axis_angle_xy(points_base)
            if aspect_ratio >= _PCA_ASPECT_RATIO_MIN:
                grasp_yaw = _shortest_grasp_yaw(angle_long + math.pi / 2)
                orientation = _oriented_topdown_quaternion(grasp_yaw)
                orientation = {k: round(v, 6) for k, v in orientation.items()}
                oriented = True
            else:
                grasp_yaw = 0.0
                orientation = TOP_DOWN_ORIENTATION
                oriented = False

            grasp_x, grasp_y, grasp_xy_method = _grasp_xy(
                points_base, (translation["x"], translation["y"])
            )
            grasp_z, pre_grasp_z, grasp_z_method = _grasp_z(points_base)
            grasp_pose = {
                "frame_id": "base_footprint",
                "position": {
                    "x": round(grasp_x, 4),
                    "y": round(grasp_y, 4),
                    "z": round(grasp_z, 4),
                },
                "orientation": orientation,
            }

            return {
                "object_name": object_name,
                "centroid_camera_frame": centroid_camera,
                "centroid_base_frame": centroid_base,
                "grasp_pose": grasp_pose,
                "grasp_xy_method": grasp_xy_method,
                "grasp_z_method": grasp_z_method,
                "pre_grasp_pose": {
                    **grasp_pose,
                    "position": {**grasp_pose["position"], "z": round(pre_grasp_z, 4)},
                },
                "bounding_box": bounding_box,
                "bbox_base_frame": bbox_base,
                "top_z": top_z,
                "num_points": len(points),
                "num_points_dropped": dropped,
                "camera_frame_id": frame_id,
                "gripper_offset_m": GRIPPER_FINGER_OFFSET_M,
                "principal_axis_angle_rad": round(float(grasp_yaw), 4),
                "principal_axis_angle_deg": round(float(grasp_yaw) * 180.0 / 3.14159265358979, 2),
                "principal_axis_aspect_ratio": round(float(aspect_ratio), 3),
                "oriented": oriented,
            }

        except (TimeoutError, RuntimeError, Exception) as e:
            # TF failed — return camera-frame data with warning
            return {
                "object_name": object_name,
                "centroid_camera_frame": centroid_camera,
                "centroid_base_frame": None,
                "grasp_pose": None,
                "bounding_box": bounding_box,
                "num_points": len(points),
                "camera_frame_id": frame_id,
                "gripper_offset_m": GRIPPER_FINGER_OFFSET_M,
                "warning": (
                    f"TF lookup failed ({str(e)}). "
                    "Centroid is in camera frame only. "
                    "Ensure tf2 buffer_server is running: "
                    "ros2 run tf2_ros buffer_server"
                ),
            }

