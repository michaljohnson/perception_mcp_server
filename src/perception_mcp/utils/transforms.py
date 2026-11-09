"""TF, rotation, and gripper-geometry helpers shared by the grasping
and placing tools.

In ROS 2 Jazzy, `LookupTransform` is an action (not a service), so we
talk to a tf2 buffer_server action via the WebSocket manager. We use a
namespaced canonical instance at `/canonical_tf/tf2_buffer_server` so
the lookup doesn't race against the five default-named buffer servers
spawned by other rclpy components (MoveIt, rviz, segmentation nodes,
etc.). See `_tf_lookup` for the rationale. The pose-computation tools
(`get_topdown_grasp_pose`, `get_topdown_placing_pose`) call `_tf_lookup`
to bring camera-frame point clouds into the robot's `base_footprint`
frame.

`TOP_DOWN_ORIENTATION` is the quaternion for a strict top-down gripper
approach: 180° rotation about X (the EEF frame has Z-up at rest, so we
flip to Z-down). Used by both grasp and drop poses.

`GRIPPER_FINGER_OFFSET_M` is the distance from the gripper's wrist
frame to the middle of the finger pads, where a grasped object sits.
MoveIt plans poses for the wrist link, so any tool computing a grasp
target needs to subtract / add this offset to convert between wrist
and finger space.
"""

import numpy as np

# Top-down approach orientation: gripper / drop pose pointing straight
# down. 180° rotation about X axis.
TOP_DOWN_ORIENTATION = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}

# Robotiq 2F-140 on the UR5: wrist flange → middle of the finger pads (gripper open),
# measured on the Summit XL: 0.22 m to the fingertips minus half the 0.065 m pad.
# When the wrist is at z = wrist_z (top-down), the pads' middle is at
# z = wrist_z - GRIPPER_FINGER_OFFSET_M.
GRIPPER_FINGER_OFFSET_M = 0.188


def _quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a unit quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def _tf_lookup(
    ws_manager,
    source_frame: str,
    target_frame: str = "base_footprint",
    timeout: float = 5.0,
) -> tuple[dict, dict]:
    """Look up a TF transform via the canonical buffer server action.

    Calls `/canonical_tf/tf2_buffer_server` (namespaced) instead of the
    default `/tf2_buffer_server` to avoid a race condition: MoveIt, rviz,
    moveit_mcp_server, the segmentation nodes, and other rclpy components
    each spawn their own `tf2_ros.Buffer.create_server()` advertising the
    default action name with independent Buffer caches. ROS 2 discovery
    routes a goal request to whichever one binds first, producing
    intermittent wrong transforms (camera-frame depth leaking through
    as base-frame z, etc.).

    The canonical instance is launched by the MCP start script under
    namespace /canonical_tf (use_sim_time true in simulation, false on the
    real robot).

    Returns:
        (translation, rotation) where translation is a dict {x, y, z}
        and rotation is a dict {x, y, z, w}. Raises on lookup failure.
    """
    result = ws_manager.send_action_goal(
        action_name="/canonical_tf/tf2_buffer_server",
        action_type="tf2_msgs/action/LookupTransform",
        goal={
            "target_frame": target_frame,
            "source_frame": source_frame,
            "source_time": {"sec": 0, "nanosec": 0},
            "timeout": {"sec": int(timeout), "nanosec": 0},
            "advanced": False,
        },
        timeout=timeout + 2.0,
    )
    transform = result.get("transform", {}).get("transform", {})
    translation = transform.get("translation", {})
    rotation = transform.get("rotation", {})
    return translation, rotation


def _transform_point(point: dict, translation: dict, rotation: dict) -> dict:
    """Transform a 3D point dict via a TF translation + quaternion rotation.

    Args:
        point: {x, y, z} in source frame.
        translation: {x, y, z} from `_tf_lookup`.
        rotation: {x, y, z, w} quaternion from `_tf_lookup`.

    Returns:
        {x, y, z} in target frame, rounded to 4 decimals.
    """
    R = _quat_to_rotation_matrix(
        rotation["x"], rotation["y"], rotation["z"], rotation["w"]
    )
    p = np.array([point["x"], point["y"], point["z"]])
    t = np.array([translation["x"], translation["y"], translation["z"]])
    out = R @ p + t
    return {
        "x": round(float(out[0]), 4),
        "y": round(float(out[1]), 4),
        "z": round(float(out[2]), 4),
    }


def _transform_points(points: np.ndarray, translation: dict, rotation: dict) -> np.ndarray:
    """Vectorised version of `_transform_point` for an Nx3 numpy array.

    Returns an Nx3 numpy array (no rounding — caller can decide).
    """
    R = _quat_to_rotation_matrix(
        rotation["x"], rotation["y"], rotation["z"], rotation["w"]
    )
    t = np.array([translation["x"], translation["y"], translation["z"]])
    return (R @ points.T).T + t


# Aspect ratio (long / short principal axis) below which the object is
# considered approximately rotationally symmetric in top-down view; for
# such shapes the grasp yaw is undefined and we fall back to the default
# top-down orientation. Empirical bands:
#   - cube / ball (true ratio ~1.0): real segmentation noise pushes
#     observed ratio to ~1.05-1.30 — bogus PCA yaw if accepted
#   - coke can on its base (true ratio ~1.0): same as cube
#   - shoe: ~2.0-3.0 — genuine long axis
#   - coke can on its side: ~2.5 — genuine long axis
#   - card deck (true ratio ~1.5): observed ~1.5
# A yaw from noise can put the IK into a wrist-twisted branch, so the
# threshold sits just above the cube noise.
_PCA_ASPECT_RATIO_MIN = 1.4


def _principal_axis_angle_xy(points_base: np.ndarray) -> tuple[float, float]:
    """Compute the principal-axis yaw and aspect ratio in the xy plane.

    Args:
        points_base: Nx3 array of points already transformed to the
            target frame (base_footprint). Only the (x, y) columns are
            used — z is dropped because grasp yaw is a planar question.

    Returns:
        ``(angle_long_rad, aspect_ratio)`` where ``angle_long_rad`` is
        the angle of the LONG axis in base xy (atan2 of the leading
        eigenvector), and ``aspect_ratio`` is ``sqrt(lambda_long /
        lambda_short)`` from the 2x2 covariance eigendecomposition.

        On degenerate input (<3 points) returns ``(0.0, 1.0)`` — the
        caller's aspect-ratio threshold will then fall back to plain
        top-down.
    """
    if points_base.shape[0] < 3:
        return 0.0, 1.0
    xy = points_base[:, :2] - points_base[:, :2].mean(axis=0)
    cov = np.cov(xy, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues; long axis is the LAST.
    lam_short, lam_long = float(eigvals[0]), float(eigvals[1])
    if lam_short <= 1e-9:
        return 0.0, 1.0
    long_axis = eigvecs[:, 1]
    angle_long = float(np.arctan2(long_axis[1], long_axis[0]))
    aspect_ratio = float(np.sqrt(lam_long / lam_short))
    return angle_long, aspect_ratio


# Below this horizontal camera-object distance the camera looks down on the
# object and sees its whole top, so the trimmed bbox middle is used instead.
_NEAR_FACE_MIN_VIEW_DIST_M = 0.10


def _grasp_xy(points_base: np.ndarray, camera_xy: tuple[float, float]) -> tuple[float, float, str]:
    """Grasp centre in the xy plane: near face + half width along the view.

    A camera sees only the side of an object that faces it, so the far half of
    the cloud is missing and a few stray points can sit behind it. The point
    mean is pulled towards the camera and the bbox middle away from it (by
    1-4 cm for small objects). The near face and the width across the view
    are stable: centre = near face + half width along the camera-object
    direction, and the middle of the width across it. Percentiles (5/95)
    ignore stray points. Assumes the object is about as deep as it is wide.

    Returns ``(x, y, method)`` with method ``"near_face"`` or ``"trimmed_bbox"``.
    """
    xy = points_base[:, :2]
    d = xy.mean(axis=0) - np.asarray(camera_xy, dtype=float)
    dist = float(np.linalg.norm(d))
    if dist < _NEAR_FACE_MIN_VIEW_DIST_M:
        lo, hi = np.percentile(xy, [5, 95], axis=0)
        mid = (lo + hi) / 2
        return float(mid[0]), float(mid[1]), "trimmed_bbox"
    d /= dist
    n = np.array([-d[1], d[0]])
    near = np.percentile(xy @ d, 5)
    lo, hi = np.percentile(xy @ n, [5, 95])
    centre = (near + (hi - lo) / 2) * d + ((lo + hi) / 2) * n
    return float(centre[0]), float(centre[1]), "near_face"


# How far an object may stick up above the finger pads without touching the
# gripper's palm (measured: more than 6 cm, kept with a margin).
_MAX_ABOVE_PADS_M = 0.05
# Wrist flange to fingertips of the Robotiq 2F-140, measured. The pad offset
# above uses 0.22; the larger value here keeps the fingertips higher.
_FINGERTIP_OFFSET_M = 0.225
# The lowest seen points sit 1-2 cm above the support (the camera rarely sees
# an object's bottom edge); the fingertips may go this far below them.
_BELOW_LOWEST_SEEN_M = 0.005
# Fingertips this far above the object's top for the pre-grasp.
_PRE_GRASP_CLEARANCE_M = 0.10


def _grasp_z(points_base: np.ndarray) -> tuple[float, float, str]:
    """Flange z for the grasp and the pre-grasp from the object's height.

    The pad middle goes to the centroid height, but no lower than
    ``_MAX_ABOVE_PADS_M`` below the object's top, so a tall object does not
    run into the palm, and never so low that the fingertips go more than
    ``_BELOW_LOWEST_SEEN_M`` below the object's lowest seen points. Top and
    bottom are the 99th and 1st z percentiles.

    Returns ``(grasp_z, pre_grasp_z, method)`` with method ``"centroid"``,
    ``"below_top"`` or ``"above_support"``.
    """
    z = points_base[:, 2]
    bottom, top = np.percentile(z, [1, 99])
    pad_z, method = float(z.mean()), "centroid"
    if pad_z < top - _MAX_ABOVE_PADS_M:
        pad_z, method = float(top - _MAX_ABOVE_PADS_M), "below_top"
    grasp_z = pad_z + GRIPPER_FINGER_OFFSET_M
    lowest = float(bottom) - _BELOW_LOWEST_SEEN_M + _FINGERTIP_OFFSET_M
    if grasp_z < lowest:
        grasp_z, method = lowest, "above_support"
    pre_grasp_z = max(grasp_z, float(top) + _FINGERTIP_OFFSET_M + _PRE_GRASP_CLEARANCE_M)
    return grasp_z, pre_grasp_z, method


def _shortest_grasp_yaw(angle_long_rad: float) -> float:
    """Wrap a long-axis angle to [-pi/2, pi/2] for minimum wrist rotation.

    The Robotiq 2F-140 is 180°-symmetric (rotating the gripper by pi
    around the approach axis produces the same physical orientation
    with fingers swapped), so any two yaws differing by pi are
    interchangeable. Choose the representative in [-pi/2, pi/2] so the
    wrist takes the shortest path from the default look_forward pose.
    """
    yaw = angle_long_rad
    while yaw > np.pi / 2:
        yaw -= np.pi
    while yaw < -np.pi / 2:
        yaw += np.pi
    return yaw


def _quat_multiply(q1: dict, q2: dict) -> dict:
    """Hamilton product q1 * q2 for two {x, y, z, w} quaternions.

    The resulting quaternion represents the rotation of applying q2
    first, then q1 (the convention that matches `v' = q ⊗ v ⊗ q*`).
    """
    x1, y1, z1, w1 = q1["x"], q1["y"], q1["z"], q1["w"]
    x2, y2, z2, w2 = q2["x"], q2["y"], q2["z"], q2["w"]
    return {
        "x": w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        "y": w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        "z": w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        "w": w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    }


def _oriented_topdown_quaternion(yaw_rad: float) -> dict:
    """Compose the canonical top-down flip with a base-frame yaw.

    The default `TOP_DOWN_ORIENTATION` rotates the gripper's tool frame
    180° about X (z-up → z-down, fingers along base -y). To rotate the
    fingers in the horizontal plane we apply a yaw about the BASE z
    axis AFTER the flip, so the operation in quaternion form is
    ``q_yaw * q_topdown``.

    Returns the composed {x, y, z, w} quaternion.
    """
    half = yaw_rad / 2.0
    q_yaw = {"x": 0.0, "y": 0.0, "z": float(np.sin(half)), "w": float(np.cos(half))}
    return _quat_multiply(q_yaw, TOP_DOWN_ORIENTATION)
