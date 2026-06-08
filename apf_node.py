"""
apf_node.py – Artificial Potential Field navigation node for ROS 2.

Subscriptions
-------------
/grid/occupancy          nav_msgs/OccupancyGrid
/localization/pose       geometry_msgs/PoseWithCovarianceStamped
/mission/current_goal    geometry_msgs/PoseStamped

Publication
-----------
/apf/target_vector       geometry_msgs/Twist   (linear.x = surge, angular.z = yaw)
"""

import math
import random

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid


# ── tuneable parameters (override via ROS 2 parameters) ─────────────────────
DEFAULT_K_ATT               = 1.0    # attractive gain
DEFAULT_K_REP               = 0.5    # repulsive gain
DEFAULT_D0                  = 2.0    # obstacle influence radius  [m]
DEFAULT_GRID_RESOLUTION     = 0.05   # fallback if grid.info unavailable [m/cell]
DEFAULT_MAX_SURGE           = 1.0    # maximum forward speed command [m/s]
DEFAULT_MAX_YAW             = 1.0    # maximum yaw rate command [rad/s]
DEFAULT_MIN_FORCE_THRESHOLD = 0.05   # below this → potential local minimum
DEFAULT_LOCAL_MINIMA_TICKS  = 20     # consecutive ticks before escape kick
DEFAULT_ESCAPE_PERTURB_MAG  = 0.5    # magnitude of random escape perturbation
DEFAULT_LOOP_HZ             = 10.0   # control-loop frequency [Hz]
"""Parameter values should be in a separate config file or ROS 2 parameter server, not hardcoded in the node. This would allow for easier tuning without code changes."""


# ── tiny value-type helpers ──────────────────────────────────────────────────
class Vector2D:
    """Lightweight 2-D vector; keeps the algorithm readable."""

    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.x = x
        self.y = y

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)
"""
Could numpy be used here for cleaner vector math, or is the overhead not worth it for such small vectors? 
Could also use in a utils module for reuse in other nodes.
"""


def wrap_to_pi(angle: float) -> float:
    """Wrap *angle* (radians) to the interval (−π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
"""Must keep the heading unit consistent (radians vs degrees) across the node and across the system. Could use a utility function to convert if needed."""

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ── main node ────────────────────────────────────────────────────────────────
class APFNode(Node):
    """Artificial Potential Field guidance node."""

    def __init__(self) -> None:
        super().__init__("apf_node")

        # ── declare & read ROS 2 parameters ─────────────────────────────────
        self._declare_params()

        self.k_att               = self.get_parameter("k_att").value
        self.k_rep               = self.get_parameter("k_rep").value
        self.d0                  = self.get_parameter("d0").value
        self.grid_resolution     = self.get_parameter("grid_resolution").value
        self.max_surge           = self.get_parameter("max_surge").value
        self.max_yaw             = self.get_parameter("max_yaw").value
        self.min_force_threshold = self.get_parameter("min_force_threshold").value
        self.local_minima_ticks  = int(self.get_parameter("local_minima_ticks").value)
        self.escape_perturb_mag  = self.get_parameter("escape_perturb_mag").value
        loop_hz                  = self.get_parameter("loop_hz").value

        # ── internal state ───────────────────────────────────────────────────
        self._grid: OccupancyGrid | None                     = None
        self._pose: PoseWithCovarianceStamped | None         = None
        self._goal: PoseStamped | None                       = None
        self._local_minima_counter: int                      = 0

        # ── QoS: latched / best-effort for map; reliable for pose & goal ────
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
        self.create_subscription(
            OccupancyGrid,
            "/grid/occupancy",
            self._cb_grid,
            map_qos,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/localization/pose",
            self._cb_pose,
            sensor_qos,
        )
        self.create_subscription(
            PoseStamped,
            "/mission/current_goal",
            self._cb_goal,
            10,
        )

        # ── publisher ────────────────────────────────────────────────────────
        self._pub_cmd = self.create_publisher(Twist, "/apf/target_vector", 10)

        # ── control loop timer ───────────────────────────────────────────────
        self.create_timer(1.0 / loop_hz, self._control_loop)

        self.get_logger().info("APF node started.")

    # ── parameter declaration ────────────────────────────────────────────────
    def _declare_params(self) -> None:
        self.declare_parameter("k_att",               DEFAULT_K_ATT)
        self.declare_parameter("k_rep",               DEFAULT_K_REP)
        self.declare_parameter("d0",                  DEFAULT_D0)
        self.declare_parameter("grid_resolution",     DEFAULT_GRID_RESOLUTION)
        self.declare_parameter("max_surge",           DEFAULT_MAX_SURGE)
        self.declare_parameter("max_yaw",             DEFAULT_MAX_YAW)
        self.declare_parameter("min_force_threshold", DEFAULT_MIN_FORCE_THRESHOLD)
        self.declare_parameter("local_minima_ticks",  float(DEFAULT_LOCAL_MINIMA_TICKS))
        self.declare_parameter("escape_perturb_mag",  DEFAULT_ESCAPE_PERTURB_MAG)
        self.declare_parameter("loop_hz",             DEFAULT_LOOP_HZ)

    # ── subscriber callbacks ─────────────────────────────────────────────────
    def _cb_grid(self, msg: OccupancyGrid) -> None:
        self._grid = msg

    def _cb_pose(self, msg: PoseWithCovarianceStamped) -> None:
        self._pose = msg

    def _cb_goal(self, msg: PoseStamped) -> None:
        self._goal = msg

    # ── helpers ──────────────────────────────────────────────────────────────
    def _publish_zero(self) -> None:
        """Publish a zero command and return immediately."""
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = 0.0
        self._pub_cmd.publish(cmd)

    def _yaw_from_quaternion(self, q) -> float:
        """Extract yaw (rotation about Z) from a geometry_msgs Quaternion."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ── attractive force ─────────────────────────────────────────────────────
    def _attractive_force(
        self,
        pose_x: float, pose_y: float,
        goal_x: float, goal_y: float,
    ) -> Vector2D:
        delta_x = goal_x - pose_x
        delta_y = goal_y - pose_y
        return Vector2D(
            x=self.k_att * delta_x,
            y=self.k_att * delta_y,
        )

    # ── repulsive force ──────────────────────────────────────────────────────
    def _repulsive_force(
        self,
        pose_x: float, pose_y: float,
        grid: OccupancyGrid,
    ) -> Vector2D:
        f_rep = Vector2D()

        info       = grid.info
        origin_x   = info.origin.position.x
        origin_y   = info.origin.position.y
        resolution = info.resolution if info.resolution > 0.0 else self.grid_resolution
        width      = info.width

        for idx, cell_value in enumerate(grid.data): # iterate over all cells O(n); could be optimized by only utilizing spatial indexing (e.g. R-tree)
            if cell_value <= 0:          # free or unknown → skip
                continue
            """cell_value < 0 → unknown → Unknown cells are ignored. Is this intended? Robot may drive into unexplored areas"""

            col = idx % width
            row = idx // width

            # world position of cell centre
            cell_x = origin_x + (col + 0.5) * resolution
            cell_y = origin_y + (row + 0.5) * resolution

            dx = pose_x - cell_x
            dy = pose_y - cell_y
            d  = math.sqrt(dx * dx + dy * dy)

            if d < 1e-3 or d > self.d0:  # too close (singularity) or outside radius
                continue
            """Singularity avoidance removes repulsion from extremely close obstacles, which may cause the robot to collide with them. Is this intended?"""

            magnitude = self.k_rep * (1.0 / d - 1.0 / self.d0) / (d * d)

            f_rep.x += magnitude * (dx / d)
            f_rep.y += magnitude * (dy / d)

        return f_rep

    # ── local-minima escape ──────────────────────────────────────────────────
    def _apply_escape_perturbation(self, f_total: Vector2D) -> Vector2D:
        random_angle = random.uniform(0.0, 2.0 * math.pi)
        f_total.x += self.escape_perturb_mag * math.cos(random_angle)
        f_total.y += self.escape_perturb_mag * math.sin(random_angle)
        return f_total

    # ── main control loop ─────────────────────────────────────────────────────
    def _control_loop(self) -> None:
        # ── guard: must have pose ────────────────────────────────────────────
        if self._pose is None:
            self.get_logger().warn("Waiting for pose …", throttle_duration_sec=5.0)
            self._publish_zero()
            return

        # ── guard: no active goal ────────────────────────────────────────────
        if self._goal is None:
            self._publish_zero()
            return

        # ── unpack pose ──────────────────────────────────────────────────────
        p        = self._pose.pose.pose
        pose_x   = p.position.x
        pose_y   = p.position.y
        theta    = self._yaw_from_quaternion(p.orientation)

        # ── unpack goal ──────────────────────────────────────────────────────
        goal_x = self._goal.pose.position.x
        goal_y = self._goal.pose.position.y
        """
        No frame validation is performed on the pose, goal or occupancy grid. Is this intended? Mismatched frames could cause erratic behavior. 
        Could use tf2 to transform everything into a common frame (e.g. "map") before processing.
        """

        # ── attractive force ─────────────────────────────────────────────────
        f_att = self._attractive_force(pose_x, pose_y, goal_x, goal_y)

        # ── repulsive force ──────────────────────────────────────────────────
        f_rep = Vector2D()
        if self._grid is not None:
            f_rep = self._repulsive_force(pose_x, pose_y, self._grid)
        else:
            self.get_logger().warn("No occupancy grid yet; repulsion disabled.",
                                   throttle_duration_sec=5.0)

        # ── total force ──────────────────────────────────────────────────────
        f_total    = f_att + f_rep
        f_magnitude = f_total.magnitude

        # ── local-minima detection & escape ──────────────────────────────────
        if f_magnitude < self.min_force_threshold:
            self._local_minima_counter += 1
        else:
            self._local_minima_counter = 0

        if self._local_minima_counter >= self.local_minima_ticks:
            self.get_logger().warn("Local minimum detected – injecting perturbation.")
            f_total     = self._apply_escape_perturbation(f_total)
            f_magnitude  = f_total.magnitude
            self._local_minima_counter = 0
            
        """
        Small force does not necessarily indicate a local minimum, e.g. when the robot is close to the goal. Is this intended?
        Suggested fix: only trigger local minima escape if the robot is not within a certain distance to the goal, e.g. by adding the following check before incrementing the local minima counter:
            if distance_to_goal < tolerance:
                stop
        """

        # ── force vector → surge / yaw ────────────────────────────────────────
        desired_heading = math.atan2(f_total.y, f_total.x)
        heading_error   = wrap_to_pi(desired_heading - theta)

        surge = clamp(f_magnitude * math.cos(heading_error), 0.0, self.max_surge)
        """Large heading errors can stall forward progress. Is this intended?"""
        yaw   = clamp(heading_error, -self.max_yaw, self.max_yaw)

        # ── publish ───────────────────────────────────────────────────────────
        cmd           = Twist()
        cmd.linear.x  = surge
        cmd.angular.z = yaw
        self._pub_cmd.publish(cmd)

        self.get_logger().debug(
            f"APF → surge={surge:.3f} m/s  yaw={yaw:.3f} rad/s  "
            f"|F|={f_magnitude:.3f}  lm_ticks={self._local_minima_counter}"
        )
        """Additional observability would simplify debugging in telemetry"""


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