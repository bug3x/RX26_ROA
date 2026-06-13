"""
apf_node.py – Artificial Potential Field navigation node for ROS 2.

Subscriptions
-------------
/grid/occupancy          nav_msgs/OccupancyGrid
/localization/pose       geometry_msgs/PoseWithCovarianceStamped
/mission/current_goal    geometry_msgs/PoseStamped

Publications
------------
/apf/target_vector       geometry_msgs/Twist        (linear.x = surge, angular.z = yaw)
/apf/debug               geometry_msgs/Vector3Stamped  (x = |F|, y = lm_ticks, z = d_goal)

Notes
-----
* All topics are expected to be in the same fixed frame (default "map").
  If the stack uses different frames, set the `fixed_frame` parameter and
  ensure tf2 can transform pose / goal into it.
* Unknown cells (value == -1) are treated as free by default.  Set
  `treat_unknown_as_occupied: true` to repel from them instead.
* Parameters live in config/apf_params.yaml; defaults here are last-resort
  fallbacks only.
"""

import math
import random
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

import tf2_ros
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    Twist,
    Vector3Stamped,
)
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid


# ── last-resort defaults (prefer config/apf_params.yaml) ────────────────────
DEFAULT_K_ATT                  = 1.0
DEFAULT_K_REP                  = 0.5
DEFAULT_D0                     = 2.0
DEFAULT_GRID_RESOLUTION        = 0.05   # [m/cell] – used only if grid.info is bad
DEFAULT_MAX_SURGE              = 1.0    # [m/s]
DEFAULT_MAX_YAW                = 1.0    # [rad/s]
DEFAULT_MIN_FORCE_THRESHOLD    = 0.05
DEFAULT_LOCAL_MINIMA_TICKS     = 20
DEFAULT_ESCAPE_PERTURB_MAG     = 0.5
DEFAULT_LOOP_HZ                = 10.0
DEFAULT_GOAL_TOLERANCE         = 0.2   # [m] – within this distance, declare arrival
DEFAULT_FIXED_FRAME            = "map"
DEFAULT_TREAT_UNKNOWN_OCCUPIED = False
"""
There will be live object detection through the computer vision and LiDAR fusion node, so unknown values in the occupancy grid should be treated as free space until the object detection can classify them as occupied.
"""


# ── helpers (reusable; consider moving to a shared utils module) ─────────────

@dataclass
class Vector2D:
    """Thin 2-D vector backed by numpy for clean math."""
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    @property
    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


def wrap_to_pi(angle: float) -> float:
    """Wrap *angle* (radians) to (−π, π]. All angles in this node are radians."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def yaw_from_quaternion(q) -> float:
    """Extract yaw from a geometry_msgs/Quaternion (rotation about Z)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# ── node ─────────────────────────────────────────────────────────────────────

class APFNode(Node):
    """Artificial Potential Field guidance node."""

    def __init__(self) -> None:
        super().__init__("apf_node")

        self._declare_params()
        self._load_params()

        # ── internal state ───────────────────────────────────────────────────
        self._grid: OccupancyGrid | None             = None
        self._pose: PoseWithCovarianceStamped | None = None
        self._goal: PoseStamped | None               = None
        self._local_minima_counter: int              = 0
        self._goal_reached_published: bool           = False

        # ── tf2 buffer for frame transforms ─────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── QoS profiles ────────────────────────────────────────────────────
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── subscriptions ────────────────────────────────────────────────────
        self.create_subscription(OccupancyGrid,
                                 "/grid/occupancy",    self._cb_grid, map_qos)
        self.create_subscription(PoseWithCovarianceStamped,
                                 "/localization/pose", self._cb_pose, sensor_qos)
        self.create_subscription(PoseStamped,
                                 "/mission/current_goal", self._cb_goal, 10)

        # ── publishers ───────────────────────────────────────────────────────
        self._pub_cmd   = self.create_publisher(Twist,          "/apf/target_vector", 10)
        self._pub_debug = self.create_publisher(Vector3Stamped, "/apf/debug",         10)
        self._pub_goal_reached = self.create_publisher(Bool, "/apf/goal_reached", 10)

        # ── control-loop timer ───────────────────────────────────────────────
        self.create_timer(1.0 / self._loop_hz, self._control_loop)

        self.get_logger().info(
            f"APF node started  frame='{self._fixed_frame}'  "
            f"unknown_as_occupied={self._treat_unknown_as_occupied}"
        )

    # ── parameter declaration ────────────────────────────────────────────────
    def _declare_params(self) -> None:
        self.declare_parameter("k_att",                   DEFAULT_K_ATT)
        self.declare_parameter("k_rep",                   DEFAULT_K_REP)
        self.declare_parameter("d0",                      DEFAULT_D0)
        self.declare_parameter("grid_resolution",         DEFAULT_GRID_RESOLUTION)
        self.declare_parameter("max_surge",               DEFAULT_MAX_SURGE)
        self.declare_parameter("max_yaw",                 DEFAULT_MAX_YAW)
        self.declare_parameter("min_force_threshold",     DEFAULT_MIN_FORCE_THRESHOLD)
        self.declare_parameter("local_minima_ticks",      float(DEFAULT_LOCAL_MINIMA_TICKS))
        self.declare_parameter("escape_perturb_mag",      DEFAULT_ESCAPE_PERTURB_MAG)
        self.declare_parameter("loop_hz",                 DEFAULT_LOOP_HZ)
        self.declare_parameter("goal_tolerance",          DEFAULT_GOAL_TOLERANCE)
        self.declare_parameter("fixed_frame",             DEFAULT_FIXED_FRAME)
        self.declare_parameter("treat_unknown_as_occupied", DEFAULT_TREAT_UNKNOWN_OCCUPIED)

    def _load_params(self) -> None:
        gp = self.get_parameter
        self._k_att                   = gp("k_att").value
        self._k_rep                   = gp("k_rep").value
        self._d0                      = gp("d0").value
        self._grid_resolution         = gp("grid_resolution").value
        self._max_surge               = gp("max_surge").value
        self._max_yaw                 = gp("max_yaw").value
        self._min_force_threshold     = gp("min_force_threshold").value
        self._local_minima_ticks      = int(gp("local_minima_ticks").value)
        self._escape_perturb_mag      = gp("escape_perturb_mag").value
        self._loop_hz                 = gp("loop_hz").value
        self._goal_tolerance          = gp("goal_tolerance").value
        self._fixed_frame             = gp("fixed_frame").value
        self._treat_unknown_as_occupied = gp("treat_unknown_as_occupied").value

    # ── subscriber callbacks ─────────────────────────────────────────────────
    def _cb_grid(self, msg: OccupancyGrid) -> None:
        self._grid = msg

    def _cb_pose(self, msg: PoseWithCovarianceStamped) -> None:
        self._pose = msg

    def _cb_goal(self, msg: PoseStamped) -> None:
        self._goal = msg
        self._local_minima_counter = 0   # reset on new goal
        self._goal_reached_published = False   # reset for new goal


    # ── frame helpers ─────────────────────────────────────────────────────────
    def _transform_pose_to_fixed(
        self,
        msg: PoseWithCovarianceStamped,
    ) -> tuple[float, float, float] | None:
        """
        Return (x, y, yaw) of the pose in self._fixed_frame.
        Returns None if the transform is unavailable.
        """
        src = msg.header.frame_id
        if src == self._fixed_frame:
            p     = msg.pose.pose
            return p.position.x, p.position.y, yaw_from_quaternion(p.orientation)
        try:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose   = msg.pose.pose
            out = self._tf_buffer.transform(ps, self._fixed_frame,
                                            timeout=rclpy.duration.Duration(seconds=0.05))
            return (
                out.pose.position.x,
                out.pose.position.y,
                yaw_from_quaternion(out.pose.orientation),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f"Cannot transform pose from '{src}' to '{self._fixed_frame}': {exc}",
                throttle_duration_sec=5.0,
            )
            return None

    def _transform_goal_to_fixed(
        self, msg: PoseStamped
    ) -> tuple[float, float] | None:
        """Return (x, y) of the goal in self._fixed_frame. None on failure."""
        src = msg.header.frame_id
        if src == self._fixed_frame:
            return msg.pose.position.x, msg.pose.position.y
        try:
            out = self._tf_buffer.transform(msg, self._fixed_frame,
                                            timeout=rclpy.duration.Duration(seconds=0.05))
            return out.pose.position.x, out.pose.position.y
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f"Cannot transform goal from '{src}' to '{self._fixed_frame}': {exc}",
                throttle_duration_sec=5.0,
            )
            return None

    # ── force calculations ────────────────────────────────────────────────────
    def _attractive_force(
        self, pose_x: float, pose_y: float,
        goal_x: float, goal_y: float,
    ) -> Vector2D:
        return Vector2D(
            x=self._k_att * (goal_x - pose_x),
            y=self._k_att * (goal_y - pose_y),
        )

    def _repulsive_force(
        self, pose_x: float, pose_y: float,
        grid: OccupancyGrid,
    ) -> Vector2D:
        """
        Compute repulsive force using numpy for efficiency.

        Performance note: iterating every cell is O(W·H). For large maps,
        consider a spatial index (e.g. scipy.spatial.cKDTree built once per
        grid update) to query only cells within d0 of the robot.
        """
        info       = grid.info
        resolution = info.resolution if info.resolution > 0.0 else self._grid_resolution
        origin_x   = info.origin.position.x
        origin_y   = info.origin.position.y
        width      = info.width
        height     = info.height

        data = np.asarray(grid.data, dtype=np.int8).reshape((height, width))

        # Occupied mask; optionally include unknown cells (value == -1)
        if self._treat_unknown_as_occupied:
            occupied_mask = data != 0          # -1 and positive values
        else:
            occupied_mask = data > 0           # strictly occupied only

        rows, cols = np.where(occupied_mask)
        if rows.size == 0:
            return Vector2D()

        cell_x = origin_x + (cols + 0.5) * resolution
        cell_y = origin_y + (rows + 0.5) * resolution

        dx = pose_x - cell_x
        dy = pose_y - cell_y
        d  = np.hypot(dx, dy)

        # Filter: inside influence radius, not degenerate
        in_range = (d >= 1e-3) & (d <= self._d0)
        dx, dy, d = dx[in_range], dy[in_range], d[in_range]

        if d.size == 0:
            return Vector2D()

        mag = self._k_rep * (1.0 / d - 1.0 / self._d0) / (d ** 2)
        # Unit vector from obstacle toward robot, scaled by magnitude
        fx = np.sum(mag * dx / d)
        fy = np.sum(mag * dy / d)
        return Vector2D(x=float(fx), y=float(fy))

    # ── local-minima escape ──────────────────────────────────────────────────
    def _apply_escape_perturbation(self, f: Vector2D) -> Vector2D:
        angle = random.uniform(0.0, 2.0 * math.pi)
        f.x  += self._escape_perturb_mag * math.cos(angle)
        f.y  += self._escape_perturb_mag * math.sin(angle)
        return f

    # ── zero command ─────────────────────────────────────────────────────────
    def _publish_zero(self) -> None:
        self._pub_cmd.publish(Twist())

    # ── debug publisher ──────────────────────────────────────────────────────
    def _publish_debug(self, f_mag: float, d_goal: float) -> None:
        msg        = Vector3Stamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._fixed_frame
        msg.vector.x = f_mag
        msg.vector.y = float(self._local_minima_counter)
        msg.vector.z = d_goal
        self._pub_debug.publish(msg)
        
    def _publish_goal_reached(self) -> None:
        if self._goal_reached_published:
            return
        self.get_logger().info("Goal reached.")
        msg = Bool()
        msg.data = True
        self._pub_goal_reached.publish(msg)
        self._goal_reached_published = True


    # ── control loop ─────────────────────────────────────────────────────────
    def _control_loop(self) -> None:

        # ── guard: pose ──────────────────────────────────────────────────────
        if self._pose is None:
            self.get_logger().warn("Waiting for pose …", throttle_duration_sec=5.0)
            self._publish_zero()
            return

        # ── guard: goal ──────────────────────────────────────────────────────
        if self._goal is None:
            self._publish_zero()
            return

        # ── frame validation & transform ─────────────────────────────────────
        pose_xyz = self._transform_pose_to_fixed(self._pose)
        if pose_xyz is None:
            self._publish_zero()
            return
        pose_x, pose_y, theta = pose_xyz

        goal_xy = self._transform_goal_to_fixed(self._goal)
        if goal_xy is None:
            self._publish_zero()
            return
        goal_x, goal_y = goal_xy

        # ── goal-reached check (suppresses false local-minima near target) ───
        d_goal = math.hypot(goal_x - pose_x, goal_y - pose_y)
        if d_goal < self._goal_tolerance:
            self._publish_goal_reached()
            self._publish_zero()
            self._local_minima_counter = 0
            return

        # ── attractive force ─────────────────────────────────────────────────
        f_att = self._attractive_force(pose_x, pose_y, goal_x, goal_y)

        # ── repulsive force ──────────────────────────────────────────────────
        if self._grid is not None:
            f_rep = self._repulsive_force(pose_x, pose_y, self._grid)
        else:
            f_rep = Vector2D()
            self.get_logger().warn("No occupancy grid yet; repulsion disabled.",
                                   throttle_duration_sec=5.0)

        # ── total force ──────────────────────────────────────────────────────
        f_total   = f_att + f_rep
        f_mag     = f_total.magnitude

        # ── local-minima detection & escape ──────────────────────────────────
        # Only triggered when far from goal; small force near goal = arrival.
        if f_mag < self._min_force_threshold:
            self._local_minima_counter += 1
        else:
            self._local_minima_counter = 0

        if self._local_minima_counter >= self._local_minima_ticks:
            self.get_logger().warn("Local minimum – injecting escape perturbation.")
            f_total = self._apply_escape_perturbation(f_total)
            f_mag   = f_total.magnitude
            self._local_minima_counter = 0

        # ── force → surge / yaw ──────────────────────────────────────────────
        desired_heading = math.atan2(f_total.y, f_total.x)
        heading_error   = wrap_to_pi(desired_heading - theta)

        # Surge scales with alignment; reduced when pointing away from target.
        # Yaw commands rotate vessel toward desired heading first.
        surge = clamp(f_mag * math.cos(heading_error), 0.0, self._max_surge)
        yaw   = clamp(heading_error, -self._max_yaw, self._max_yaw)

        # ── publish command ───────────────────────────────────────────────────
        cmd           = Twist()
        cmd.linear.x  = surge
        cmd.angular.z = yaw
        self._pub_cmd.publish(cmd)

        # ── publish debug telemetry ───────────────────────────────────────────
        self._publish_debug(f_mag, d_goal)

        self.get_logger().debug(
            f"surge={surge:.3f} yaw={yaw:.3f} |F|={f_mag:.3f} "
            f"d_goal={d_goal:.3f} lm={self._local_minima_counter}"
        )


# ── entry-point ──────────────────────────────────────────────────────────────
def main(args=None) -> None:
    rclpy.init(args=args)
    node = APFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()