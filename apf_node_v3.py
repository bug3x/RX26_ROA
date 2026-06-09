"""
apf_node.py – Artificial Potential Field navigation node for ROS 2.

Subscriptions
-------------
/grid/occupancy          nav_msgs/OccupancyGrid
/localization/pose       geometry_msgs/PoseWithCovarianceStamped
/mission/current_goal    geometry_msgs/PoseStamped

Publications
------------
/apf/target_vector       geometry_msgs/Twist
                             linear.x  = surge [m/s]
                             angular.z = yaw   [rad/s]
/apf/debug               geometry_msgs/TwistStamped
                             linear.x  = |F_total|
                             linear.y  = |F_att|
                             linear.z  = |F_rep|
                             angular.x = lm_progress_deficit [m]
                             angular.y = d_goal [m]
                             angular.z = local_minima_counter
/apf/goal_reached        std_msgs/Bool  (latched; True while inside goal_tolerance)

Notes
-----
* All topics are expected to be in the same fixed frame (default "map").
  Set the `fixed_frame` parameter and ensure tf2 can transform if frames differ.
* Unknown cells (value == -1) are treated as free by default. Set
  `treat_unknown_as_occupied: true` to repel from them.
* Parameters live in config/apf_params.yaml; in-code defaults are last-resort
  fallbacks only.
* Local-minima detection uses progress tracking: a minimum is declared when
  the robot has not closed distance to the goal by `lm_progress_min_dist` [m]
  within `lm_progress_window` seconds, rather than a raw force-magnitude check.
* Dynamic obstacles, DWA / velocity-obstacle integration, and multi-goal
  sequencing are intentionally out of scope for this node.
"""

import math
import random
from dataclasses import dataclass, field

import numpy as np
import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 – registers PoseStamped transform support
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    Twist,
    TwistStamped,
)
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, ParameterType
from std_msgs.msg import Bool


# ── last-resort defaults (prefer config/apf_params.yaml) ────────────────────
DEFAULT_K_ATT                   = 1.0
DEFAULT_K_REP                   = 0.5
DEFAULT_D0                      = 2.0
DEFAULT_GRID_RESOLUTION         = 0.05   # [m/cell] – fallback only
DEFAULT_MAX_SURGE               = 1.0    # [m/s]
DEFAULT_MAX_YAW                 = 1.0    # [rad/s]
DEFAULT_ESCAPE_PERTURB_MAG      = 0.5
DEFAULT_LOOP_HZ                 = 10.0
DEFAULT_GOAL_TOLERANCE          = 0.2    # [m]
DEFAULT_FIXED_FRAME             = "map"
DEFAULT_TREAT_UNKNOWN_OCCUPIED  = False
DEFAULT_REPULSION_ENABLED       = True
DEFAULT_LM_PROGRESS_WINDOW      = 5.0   # [s]  look-back window for progress check
DEFAULT_LM_PROGRESS_MIN_DIST    = 0.1   # [m]  minimum progress required in window


# ── helpers ──────────────────────────────────────────────────────────────────

@dataclass
class Vector2D:
    """Lightweight 2-D vector with in-place addition and numpy interop."""
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __iadd__(self, other: "Vector2D") -> "Vector2D":
        self.x += other.x
        self.y += other.y
        return self

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


def _fp_descriptor(description: str, low: float, high: float,
                   step: float = 0.0) -> ParameterDescriptor:
    """Build a ParameterDescriptor with a floating-point range for bounds checking."""
    rng = FloatingPointRange(from_value=low, to_value=high, step=step)
    return ParameterDescriptor(
        description=description,
        type=ParameterType.PARAMETER_DOUBLE,
        floating_point_range=[rng],
    )


# ── node ─────────────────────────────────────────────────────────────────────

class APFNode(Node):
    """Artificial Potential Field guidance node."""

    def __init__(self) -> None:
        super().__init__("apf_node")

        self._declare_params()
        self._load_params()

        # ── internal state ───────────────────────────────────────────────────
        self._grid: OccupancyGrid | None              = None
        self._grid_cache: tuple | None                = None  # (seq, occ_xy arrays)
        self._pose: PoseWithCovarianceStamped | None  = None
        self._goal: PoseStamped | None                = None

        # progress-based local-minima tracking
        # stores (timestamp_sec, d_goal) snapshots over the look-back window
        # using a list of tuples, as there may be duplicate timestamps
        self._progress_history: list[tuple[float, float]] = []
        self._local_minima_counter: int = 0

        # goal-reached latch
        self._goal_reached_published: bool = False

        # ── tf2 ──────────────────────────────────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── QoS ─────────────────────────────────────────────────────────────
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── subscriptions ────────────────────────────────────────────────────
        self.create_subscription(OccupancyGrid,
                                 "/grid/occupancy",       self._cb_grid, map_qos)
        self.create_subscription(PoseWithCovarianceStamped,
                                 "/localization/pose",    self._cb_pose,
                                 qos_profile_sensor_data)
        self.create_subscription(PoseStamped,
                                 "/mission/current_goal", self._cb_goal, 10)

        # ── publishers ───────────────────────────────────────────────────────
        self._pub_cmd          = self.create_publisher(Twist,        "/apf/target_vector", 10)
        self._pub_debug        = self.create_publisher(TwistStamped, "/apf/debug",         10)
        self._pub_goal_reached = self.create_publisher(Bool,         "/apf/goal_reached",  10)

        # ── timer ────────────────────────────────────────────────────────────
        self.create_timer(1.0 / self._loop_hz, self._control_loop)

        self.get_logger().info(
            f"APF node started  frame='{self._fixed_frame}'  "
            f"repulsion={'ON' if self._repulsion_enabled else 'OFF'}  "
            f"unknown_as_occupied={self._treat_unknown_as_occupied}"
        )

    # ── parameter declaration with bounds ────────────────────────────────────
    def _declare_params(self) -> None:
        self.declare_parameter(
            "k_att", DEFAULT_K_ATT,
            _fp_descriptor("Attractive gain (must be > 0)", 1e-6, 100.0))
        self.declare_parameter(
            "k_rep", DEFAULT_K_REP,
            _fp_descriptor("Repulsive gain (must be > 0)", 1e-6, 100.0))
        self.declare_parameter(
            "d0", DEFAULT_D0,
            _fp_descriptor("Obstacle influence radius [m] (must be > 0)", 0.01, 50.0))
        self.declare_parameter(
            "grid_resolution", DEFAULT_GRID_RESOLUTION,
            _fp_descriptor("Fallback grid resolution [m/cell]", 0.001, 10.0))
        self.declare_parameter(
            "max_surge", DEFAULT_MAX_SURGE,
            _fp_descriptor("Maximum forward speed [m/s]", 0.0, 20.0))
        self.declare_parameter(
            "max_yaw", DEFAULT_MAX_YAW,
            _fp_descriptor("Maximum yaw rate [rad/s]", 0.0, 2 * math.pi))
        self.declare_parameter(
            "escape_perturb_mag", DEFAULT_ESCAPE_PERTURB_MAG,
            _fp_descriptor("Escape perturbation magnitude", 0.0, 10.0))
        self.declare_parameter(
            "loop_hz", DEFAULT_LOOP_HZ,
            _fp_descriptor("Control loop frequency [Hz]", 0.1, 100.0))
        self.declare_parameter(
            "goal_tolerance", DEFAULT_GOAL_TOLERANCE,
            _fp_descriptor("Goal arrival radius [m]", 0.01, 100.0))
        self.declare_parameter(
            "lm_progress_window", DEFAULT_LM_PROGRESS_WINDOW,
            _fp_descriptor(
                "Look-back window [s] for progress-based local-minima detection",
                0.5, 120.0))
        self.declare_parameter(
            "lm_progress_min_dist", DEFAULT_LM_PROGRESS_MIN_DIST,
            _fp_descriptor(
                "Minimum goal-distance reduction [m] required within the window "
                "before a local minimum is declared",
                0.001, 10.0))
        self.declare_parameter("fixed_frame",             DEFAULT_FIXED_FRAME)
        self.declare_parameter("treat_unknown_as_occupied", DEFAULT_TREAT_UNKNOWN_OCCUPIED)
        self.declare_parameter("repulsion_enabled",         DEFAULT_REPULSION_ENABLED)

    def _load_params(self) -> None:
        gp = self.get_parameter
        self._k_att                     = gp("k_att").value
        self._k_rep                     = gp("k_rep").value
        self._d0                        = gp("d0").value
        self._grid_resolution           = gp("grid_resolution").value
        self._max_surge                 = gp("max_surge").value
        self._max_yaw                   = gp("max_yaw").value
        self._escape_perturb_mag        = gp("escape_perturb_mag").value
        self._loop_hz                   = gp("loop_hz").value
        self._goal_tolerance            = gp("goal_tolerance").value
        self._lm_progress_window        = gp("lm_progress_window").value
        self._lm_progress_min_dist      = gp("lm_progress_min_dist").value
        self._fixed_frame               = gp("fixed_frame").value
        self._treat_unknown_as_occupied = gp("treat_unknown_as_occupied").value
        self._repulsion_enabled         = gp("repulsion_enabled").value

    # ── subscriber callbacks ─────────────────────────────────────────────────
    def _cb_grid(self, msg: OccupancyGrid) -> None:
        self._grid = msg
        self._grid_cache = None          # invalidate cache on new grid

    def _cb_pose(self, msg: PoseWithCovarianceStamped) -> None:
        self._pose = msg

    def _cb_goal(self, msg: PoseStamped) -> None:
        self._goal = msg
        self._progress_history.clear()
        self._local_minima_counter  = 0
        self._goal_reached_published = False

    # ── frame helpers ─────────────────────────────────────────────────────────
    def _transform_pose_to_fixed(
        self, msg: PoseWithCovarianceStamped,
    ) -> tuple[float, float, float] | None:
        """Return (x, y, yaw) in self._fixed_frame, or None on failure."""
        src = msg.header.frame_id
        if src == self._fixed_frame:
            p = msg.pose.pose
            return p.position.x, p.position.y, yaw_from_quaternion(p.orientation)
        try:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose   = msg.pose.pose
            out = self._tf_buffer.transform(
                ps, self._fixed_frame,
                timeout=rclpy.duration.Duration(seconds=0.05))
            return (out.pose.position.x, out.pose.position.y,
                    yaw_from_quaternion(out.pose.orientation))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f"Cannot transform pose from '{src}' to '{self._fixed_frame}': {exc}",
                throttle_duration_sec=5.0)
            return None

    def _transform_goal_to_fixed(
        self, msg: PoseStamped,
    ) -> tuple[float, float] | None:
        """Return (x, y) of goal in self._fixed_frame, or None on failure."""
        src = msg.header.frame_id
        if src == self._fixed_frame:
            return msg.pose.position.x, msg.pose.position.y
        try:
            out = self._tf_buffer.transform(
                msg, self._fixed_frame,
                timeout=rclpy.duration.Duration(seconds=0.05))
            return out.pose.position.x, out.pose.position.y
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f"Cannot transform goal from '{src}' to '{self._fixed_frame}': {exc}",
                throttle_duration_sec=5.0)
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

    def _get_occupied_cell_coords(self, grid: OccupancyGrid) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (cell_x, cell_y) world-coordinate arrays for all occupied cells.

        Result is cached by grid sequence number and reused across control ticks
        until a new grid message arrives — grids update far slower than the loop.
        """
        seq = id(grid)   # object identity as a lightweight change key
        if self._grid_cache is not None and self._grid_cache[0] == seq:
            return self._grid_cache[1], self._grid_cache[2]

        info       = grid.info
        resolution = info.resolution if info.resolution > 0.0 else self._grid_resolution
        origin_x   = info.origin.position.x
        origin_y   = info.origin.position.y
        width      = info.width
        height     = info.height

        data = np.frombuffer(bytes(grid.data), dtype=np.int8).reshape((height, width))

        if self._treat_unknown_as_occupied:
            occupied_mask = data != 0
        else:
            occupied_mask = data > 0

        rows, cols = np.where(occupied_mask)
        cell_x = origin_x + (cols + 0.5) * resolution
        cell_y = origin_y + (rows + 0.5) * resolution

        self._grid_cache = (seq, cell_x, cell_y)
        return cell_x, cell_y

    def _repulsive_force(
        self, pose_x: float, pose_y: float,
        grid: OccupancyGrid,
    ) -> Vector2D:
        """
        Repulsive force via vectorised numpy over cached occupied-cell coordinates.

        Time:  O(W·H) on cache miss (grid update), O(K) on cache hit (K = occupied cells).
        Space: O(K) for the in-range subset arrays.

        For large maps consider a cKDTree spatial index (see engineering doc §5.1).
        """
        cell_x, cell_y = self._get_occupied_cell_coords(grid)
        if cell_x.size == 0:
            return Vector2D()

        dx = pose_x - cell_x
        dy = pose_y - cell_y
        d  = np.hypot(dx, dy)

        in_range = (d >= 1e-3) & (d <= self._d0)
        dx, dy, d = dx[in_range], dy[in_range], d[in_range]
        if d.size == 0:
            return Vector2D()

        mag = self._k_rep * (1.0 / d - 1.0 / self._d0) / (d ** 2)
        return Vector2D(x=float(np.sum(mag * dx / d)),
                        y=float(np.sum(mag * dy / d)))

    # ── local-minima detection (progress-based) ───────────────────────────────
    def _update_local_minima(self, now_sec: float, d_goal: float) -> bool:
        """
        Return True if the robot is stuck in a local minimum.

        A minimum is declared when the closest d_goal recorded in the last
        `lm_progress_window` seconds has not improved by at least
        `lm_progress_min_dist` metres compared to the oldest sample in the window.

        This avoids false positives when the robot is simply turning in place or
        moving slowly — conditions where force-magnitude-only detection fires incorrectly.
        """
        self._progress_history.append((now_sec, d_goal))

        # drop samples older than the window
        cutoff = now_sec - self._lm_progress_window
        self._progress_history = [
            (t, d) for t, d in self._progress_history if t >= cutoff
        ]

        if len(self._progress_history) < 2:
            return False

        oldest_d = self._progress_history[0][1]
        best_d   = min(d for _, d in self._progress_history)
        progress = oldest_d - best_d   # positive = robot is getting closer

        stuck = progress < self._lm_progress_min_dist
        if stuck:
            self._local_minima_counter += 1
        else:
            self._local_minima_counter = 0
        return stuck

    # ── local-minima escape ───────────────────────────────────────────────────
    def _apply_escape_perturbation(self, f: Vector2D) -> Vector2D:
        angle = random.uniform(0.0, 2.0 * math.pi)
        f.x  += self._escape_perturb_mag * math.cos(angle)
        f.y  += self._escape_perturb_mag * math.sin(angle)
        return f

    # ── publishers ────────────────────────────────────────────────────────────
    def _publish_zero(self) -> None:
        self._pub_cmd.publish(Twist())

    def _publish_goal_reached(self) -> None:
        """Publish Bool(True) every tick while inside goal_tolerance (latched behaviour)."""
        msg      = Bool()
        msg.data = True
        self._pub_goal_reached.publish(msg)
        if not self._goal_reached_published:
            self.get_logger().info("Goal reached.")
            self._goal_reached_published = True

    def _publish_debug(
        self,
        f_total: Vector2D, f_att: Vector2D, f_rep: Vector2D,
        d_goal: float,
    ) -> None:
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._fixed_frame
        # linear  → force magnitudes
        msg.twist.linear.x  = f_total.magnitude
        msg.twist.linear.y  = f_att.magnitude
        msg.twist.linear.z  = f_rep.magnitude
        # angular → diagnostics
        msg.twist.angular.x = float(self._progress_history[0][1] - min(d for _, d in self._progress_history)
                                    if len(self._progress_history) >= 2 else 0.0)
        msg.twist.angular.y = d_goal
        msg.twist.angular.z = float(self._local_minima_counter)
        self._pub_debug.publish(msg)

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

        # ── goal-reached check ───────────────────────────────────────────────
        d_goal = math.hypot(goal_x - pose_x, goal_y - pose_y)
        if d_goal < self._goal_tolerance:
            self._publish_goal_reached()
            self._publish_zero()
            self._progress_history.clear()
            self._local_minima_counter = 0
            return

        # ── attractive force ─────────────────────────────────────────────────
        f_att = self._attractive_force(pose_x, pose_y, goal_x, goal_y)

        # ── repulsive force ──────────────────────────────────────────────────
        if self._repulsion_enabled and self._grid is not None:
            f_rep = self._repulsive_force(pose_x, pose_y, self._grid)
        else:
            f_rep = Vector2D()
            if not self._repulsion_enabled:
                self.get_logger().warn("Repulsion disabled (test mode).",
                                       throttle_duration_sec=10.0)
            elif self._grid is None:
                self.get_logger().warn("No occupancy grid; repulsion inactive.",
                                       throttle_duration_sec=5.0)

        # ── total force ──────────────────────────────────────────────────────
        f_total = f_att + f_rep

        # ── local-minima detection & escape ──────────────────────────────────
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self._update_local_minima(now_sec, d_goal):
            self.get_logger().warn(
                f"Local minimum – no progress ({self._lm_progress_min_dist} m) "
                f"in {self._lm_progress_window} s. Injecting escape perturbation.")
            f_total = self._apply_escape_perturbation(f_total)
            self._progress_history.clear()
            self._local_minima_counter = 0

        # ── force → surge / yaw ──────────────────────────────────────────────
        f_mag           = f_total.magnitude
        desired_heading = math.atan2(f_total.y, f_total.x)
        heading_error   = wrap_to_pi(desired_heading - theta)

        # Vessel yaws first, then surges (USV dynamics — intentional).
        surge = clamp(f_mag * math.cos(heading_error), 0.0, self._max_surge)
        yaw   = clamp(heading_error, -self._max_yaw, self._max_yaw)

        # ── publish ───────────────────────────────────────────────────────────
        cmd           = Twist()
        cmd.linear.x  = surge
        cmd.angular.z = yaw
        self._pub_cmd.publish(cmd)

        self._publish_debug(f_total, f_att, f_rep, d_goal)

        self.get_logger().debug(
            f"surge={surge:.3f} yaw={yaw:.3f} |F|={f_mag:.3f} "
            f"d_goal={d_goal:.3f} lm_ctr={self._local_minima_counter}"
        )


# ── entry-point ──────────────────────────────────────────────────────────────
def main(args: list[str] | None = None) -> None:
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