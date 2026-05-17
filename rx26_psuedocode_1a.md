# RX26 Reactive Obstacle Avoidance

================================================================================
## CONSTANTS & CONFIGURATION==
==============================================================================

```
# --- Sensor parameters ---
VOXEL_SIZE          = 0.05        # meters per voxel cube side
OUTLIER_K_NEIGHBORS = 10          # neighbors to check for outlier removal
OUTLIER_STD_RATIO   = 2.0         # points beyond mean + n*sigma are dropped
ICP_MAX_ITERATIONS  = 50
ICP_CONVERGENCE_TOL = 1e-6        # mean point distance threshold

# --- Camera intrinsics (calibrated offline) ---
FX = 600.0                        # focal length x (pixels)
FY = 600.0                        # focal length y (pixels)
CX = 640.0                        # principal point x
CY = 360.0                        # principal point y
T_lidar_to_camera = Matrix4x4     # extrinsic transform from ICP calibration

# --- Occupancy grid ---
GRID_RESOLUTION     = 0.1         # meters per cell
GRID_SIZE_M         = 20.0        # grid spans 20m x 20m around vessel
CELL_DECAY_TAU      = 5.0         # seconds until cell confidence halves
OBSTACLE_THRESHOLD  = 0.5         # confidence above this = occupied

# --- EKF ---
Q = process_noise_covariance      # 5x5 matrix (x, y, theta, v, omega)
R_gps = gps_noise_covariance      # 2x2
R_imu = imu_noise_covariance      # 3x3

# --- APF ---
K_ATT               = 1.0         # attractive gain
K_REP               = 2.0         # repulsive gain
D0                  = 2.0         # influence radius (meters)
MIN_FORCE_THRESHOLD = 0.05        # below this = local minima detected
LOCAL_MINIMA_TICKS  = 10          # ticks stuck before escape triggers
ESCAPE_PERTURB_MAG  = 0.3         # random walk magnitude during escape

# --- Low-level controller ---
KP_HEADING          = 1.2
KI_HEADING          = 0.01
KD_HEADING          = 0.3
KP_SPEED            = 0.8
MAX_SURGE           = 1.0         # m/s
MAX_YAW             = 1.0         # rad/s
WATCHDOG_TIMEOUT    = 10.0        # seconds — Teensy halts if no command

# --- Mission Manager ---
GOAL_REACHED_RADIUS = 0.5         # meters — goal considered reached
TASK_TIMEOUT        = 120.0       # seconds per task before abort
```

================================================================================
## DATA STRUCTURES
================================================================================

```
struct Point3D:
    x, y, z         : float
    label           : string       # e.g. "red_buoy", "green_buoy", ""
    confidence      : float        # YOLO confidence [0,1]

struct Pose:
    x, y            : float        # world frame position (meters)
    theta           : float        # heading (radians)
    v               : float        # linear velocity (m/s)
    omega           : float        # angular velocity (rad/s)

struct GridCell:
    occupied        : bool
    confidence      : float        # [0,1], decays over time
    label           : string
    last_updated    : timestamp

struct OccupancyGrid:
    cells           : 2D array of GridCell
    origin_x        : float        # world coords of grid [0,0]
    origin_y        : float
    resolution      : float

struct Vector2D:
    x, y            : float

struct MissionState:
    current_task    : enum {SAFE_PASSAGE, SURVEY_REPAIR, COORD_LOGISTICS, IDLE}
    current_goal    : Point2D
    task_complete   : bool
    uav_report      : list of Point2D   # buoy positions from UAV
    timeout_timer   : float
```

================================================================================
## NODE 1 — LIDAR API
## Produces raw 360° point cloud each scan cycle (~10 Hz)
================================================================================

```
FUNCTION lidar_api_node():

    LOOP forever:
        raw_scan = hardware_read_lidar()
        # raw_scan = list of (range, azimuth, elevation) in spherical coords

        point_cloud = []
        FOR each (r, az, el) in raw_scan:
            IF r < MIN_RANGE or r > MAX_RANGE:
                CONTINUE                     # discard invalid returns
            x = r * cos(el) * cos(az)
            y = r * cos(el) * sin(az)
            z = r * sin(el)
            point_cloud.APPEND(Point3D(x, y, z))

        PUBLISH /lidar/raw_cloud → point_cloud
        SLEEP until next scan cycle
```

================================================================================
## NODE 2 — CAMERA API
## Produces YOLO bounding boxes + class labels at ~30 Hz
================================================================================

```
FUNCTION camera_api_node():

    model = load_yolo_model("yolov8n_buoys.pt") # example YOLOv8 model input

    LOOP forever:
        frame = hardware_read_camera()

        detections = model.infer(frame)
        # detections = list of {bbox: (u1,v1,u2,v2), label, confidence}

        # Filter low-confidence detections
        detections = FILTER detections WHERE confidence > 0.5

        PUBLISH /camera/detections → detections
        PUBLISH /camera/frame      → frame       # for projection step
        SLEEP until next frame
```

================================================================================
## NODE 3 — GPS API
## Publishes raw lat/lon at ~5 Hz
================================================================================

```
FUNCTION gps_api_node():

    LOOP forever:
        raw = hardware_read_gps()
        # raw = {lat, lon, fix_quality, num_satellites}

        IF raw.fix_quality == NO_FIX:
            PUBLISH /gps/status → "NO_FIX"
            CONTINUE

        # Convert geodetic to local ENU (East-North-Up) cartesian
        x, y = latlon_to_ENU(raw.lat, raw.lon, origin_lat, origin_lon)

        PUBLISH /gps/position → {x, y, fix_quality}
        SLEEP until next reading
```

================================================================================
## NODE 4 — IMU API
## Publishes accelerometer + gyroscope + compass at ~100 Hz
================================================================================

```
FUNCTION imu_api_node():

    LOOP forever:
        raw = hardware_read_imu()

        # Apply calibration offsets (done once at startup)
        accel = raw.accel - accel_bias
        gyro  = raw.gyro  - gyro_bias
        mag   = raw.mag   - mag_bias

        heading = atan2(mag.y, mag.x)   # compass heading (radians)

        PUBLISH /imu/data → {accel, gyro, heading}
        SLEEP until next reading
```

================================================================================
## NODE 5 — LOCALIZATION NODE (EKF)
## Fuses GPS + IMU + COMPASS → world pose estimate
## Extended Kalman Filter implimentation
================================================================================

```
FUNCTION localization_node():

    # EKF state vector: [x, y, theta, v, omega]
    state       = [0, 0, 0, 0, 0]
    covariance  = identity(5) * 0.1       # initial uncertainty

    last_time   = now()

    LOOP forever:

        dt = now() - last_time
        last_time = now()

        # ── PREDICT STEP ────────────────────────────────────────────────
        # Motion model: constant velocity with turn rate
        #   x_new     = x + v * cos(theta) * dt
        #   y_new     = y + v * sin(theta) * dt
        #   theta_new = theta + omega * dt
        #   v_new     = v          (assumed constant between updates)
        #   omega_new = omega

        imu_data = READ /imu/data

        # Use gyro for omega (more accurate than GPS for short dt)
        state.omega = imu_data.gyro.z

        F = jacobian_of_motion_model(state, dt)   # 5x5 linearization

        state      = motion_model(state, dt)
        covariance = F * covariance * F.T + Q

        # ── UPDATE STEP — GPS ────────────────────────────────────────────
        IF new GPS reading available:
            gps = READ /gps/position

            H_gps = [[1,0,0,0,0],          # measurement maps x,y from state
                     [0,1,0,0,0]]

            innovation = [gps.x - state.x,
                          gps.y - state.y]

            S = H_gps * covariance * H_gps.T + R_gps
            K = covariance * H_gps.T * inverse(S)   # Kalman gain

            state      = state + K * innovation
            covariance = (I - K * H_gps) * covariance

        # ── UPDATE STEP — COMPASS ────────────────────────────────────────
        IF new heading reading available:
            H_mag = [[0,0,1,0,0]]           # measurement maps theta

            innovation = wrap_to_pi(imu_data.heading - state.theta)

            S = H_mag * covariance * H_mag.T + R_imu[2,2]
            K = covariance * H_mag.T * inverse(S)

            state      = state + K * [innovation]
            covariance = (I - K * H_mag) * covariance

        pose = Pose(x=state.x, y=state.y,
                    theta=state.theta,
                    v=state.v, omega=state.omega)

        PUBLISH /localization/pose → pose
        PUBLISH /localization/covariance → covariance
```

================================================================================
## NODE 6 — SENSOR FUSION
## Stage A: Voxel filter + outlier removal on LiDAR
## Stage B: ICP-based projection (pixel → point cloud mapping)
## only geometric processing, no kalman filter
================================================================================

```
FUNCTION sensor_fusion_node():

    # Load extrinsic transform (computed offline via ICP calibration)
    T_lidar_cam = load_transform("lidar_camera_extrinsic.yaml")

    LOOP forever:

        raw_cloud  = READ /lidar/raw_cloud
        detections = READ /camera/detections

        # ── STAGE A: VOXEL GRID FILTER ───────────────────────────────────

        voxel_map = empty HashMap                # key = (vx, vy, vz) voxel index

        FOR each point P in raw_cloud:
            vx = floor(P.x / VOXEL_SIZE)
            vy = floor(P.y / VOXEL_SIZE)
            vz = floor(P.z / VOXEL_SIZE)
            key = (vx, vy, vz)
            voxel_map[key].APPEND(P)

        # Replace each voxel's points with their centroid
        downsampled = []
        FOR each (key, points) in voxel_map:
            centroid = mean(points)              # average x, y, z
            downsampled.APPEND(centroid)

        # ── STAGE A: STATISTICAL OUTLIER REMOVAL ─────────────────────────

        kdtree = build_kdtree(downsampled)
        mean_dists = []

        FOR each point P in downsampled:
            neighbors = kdtree.query(P, k=OUTLIER_K_NEIGHBORS)
            mean_dists.APPEND(mean(distances_to(P, neighbors)))

        global_mean  = mean(mean_dists)
        global_sigma = std_dev(mean_dists)
        threshold    = global_mean + OUTLIER_STD_RATIO * global_sigma

        filtered_cloud = [P for P, d in zip(downsampled, mean_dists)
                          IF d < threshold]

        # ── STAGE B: CAMERA → LIDAR FRAME TRANSFORM ──────────────────────
        # Apply stored extrinsic to bring LiDAR points into camera frame
        # (this is the real-time application of the offline ICP result)

        cloud_in_cam_frame = []
        FOR each point P in filtered_cloud:
            P_cam = T_lidar_cam * [P.x, P.y, P.z, 1.0]  # homogeneous mult
            cloud_in_cam_frame.APPEND(Point3D(P_cam.x, P_cam.y, P_cam.z))

        # ── STAGE B: PIXEL → POINT PROJECTION ────────────────────────────
        # Project each 3D LiDAR point (in camera frame) to 2D pixel
        # Look up which YOLO bounding box it lands in
        # Assign that detection's label to the point

        labeled_cloud = []
        FOR each P_cam in cloud_in_cam_frame:
            IF P_cam.z <= 0:
                CONTINUE             # behind camera, skip

            u = FX * (P_cam.x / P_cam.z) + CX
            v = FY * (P_cam.y / P_cam.z) + CY

            label      = ""
            confidence = 0.0

            FOR each det in detections:
                IF u >= det.bbox.u1 AND u <= det.bbox.u2
                AND v >= det.bbox.v1 AND v <= det.bbox.v2:
                    label      = det.label
                    confidence = det.confidence
                    BREAK

            labeled_cloud.APPEND(
                Point3D(P_cam.x, P_cam.y, P_cam.z,
                        label=label, confidence=confidence)
            )

        # Output is still in BODY frame (camera frame ≈ body frame
        # after extrinsic correction). World frame transform is next.

        PUBLISH /fusion/labeled_cloud_body → labeled_cloud
```

================================================================================
## NODE 7 — COORDINATE TRANSFORM NODE
## Rotates body-frame obstacle cloud into world frame using EKF pose
================================================================================

```
FUNCTION coordinate_transform_node():

    LOOP forever:

        labeled_cloud = READ /fusion/labeled_cloud_body
        pose          = READ /localization/pose

        theta = pose.theta     # vessel heading in world frame

        # 2D rotation matrix (z-up, rotate around vertical axis)
        # | cos(θ)  -sin(θ) |
        # | sin(θ)   cos(θ) |

        world_frame_cloud = []

        FOR each P in labeled_cloud:
            # Rotate body-frame (x_b, y_b) by heading theta
            x_world = pose.x + P.x * cos(theta) - P.y * sin(theta)
            y_world = pose.y + P.x * sin(theta) + P.y * cos(theta)

            world_frame_cloud.APPEND(
                Point3D(x_world, y_world, P.z,
                        label=P.label, confidence=P.confidence)
            )

        PUBLISH /fusion/labeled_cloud_world → world_frame_cloud
```

================================================================================
## NODE 8 — OCCUPANCY GRID
## Maintains a 2D world-frame map of obstacle cells
## Includes cell aging / decay to remove stale data
================================================================================

```
FUNCTION occupancy_grid_node():

    grid = OccupancyGrid(
        cells      = 2D array[(GRID_SIZE_M/GRID_RESOLUTION)²] of GridCell,
        origin_x   = 0.0,
        origin_y   = 0.0,
        resolution = GRID_RESOLUTION
    )

    last_decay_time = now()

    LOOP forever:

        world_cloud = READ /fusion/labeled_cloud_world
        pose        = READ /localization/pose

        # Re-center grid origin around vessel to keep vessel near center
        grid.origin_x = pose.x - (GRID_SIZE_M / 2)
        grid.origin_y = pose.y - (GRID_SIZE_M / 2)

        # ── UPDATE CELLS from new detections ─────────────────────────────

        FOR each P in world_cloud:
            col = floor((P.x - grid.origin_x) / GRID_RESOLUTION)
            row = floor((P.y - grid.origin_y) / GRID_RESOLUTION)

            IF out_of_bounds(col, row):
                CONTINUE

            cell = grid.cells[row][col]
            cell.occupied     = true
            cell.confidence   = min(1.0, cell.confidence + P.confidence * 0.3)
            cell.label        = P.label IF P.confidence > cell.confidence
            cell.last_updated = now()

        # ── CELL DECAY ────────────────────────────────────────────────────
        # Cells not recently updated lose confidence over time
        # Prevents ghost obstacles from old positions

        dt_decay = now() - last_decay_time
        last_decay_time = now()

        FOR each cell in grid.cells:
            IF cell.occupied:
                # Exponential decay: confidence halves every CELL_DECAY_TAU seconds
                decay_factor   = exp(-dt_decay * ln(2) / CELL_DECAY_TAU)
                cell.confidence = cell.confidence * decay_factor

                IF cell.confidence < OBSTACLE_THRESHOLD:
                    cell.occupied = false
                    cell.label    = ""

        # ── MARK GOAL CELL (from Mission Manager) ────────────────────────

        goal = READ /mission/current_goal (non-blocking)
        IF goal is not None:
            goal_col = floor((goal.x - grid.origin_x) / GRID_RESOLUTION)
            goal_row = floor((goal.y - grid.origin_y) / GRID_RESOLUTION)
            grid.goal_cell = (goal_row, goal_col)

        PUBLISH /grid/occupancy → grid
```

================================================================================
## NODE 9 — MISSION MANAGER (State Machine)
## Determines current goal, handles task sequencing,
## subscribes to UAV reports for RobotX 2026 cross-domain handoff
================================================================================

```
FUNCTION mission_manager_node():

    state = MissionState(
        current_task = IDLE,
        current_goal = None,
        task_complete = false,
        timeout_timer = 0.0
    )

    waypoint_queue = []

    LOOP forever:

        pose = READ /localization/pose

        # ── CHECK FOR UAV HANDOFF (Task 1: Safe Passage) ─────────────────

        uav_report = READ /mission/task1/report (non-blocking)
        IF uav_report is not None:
            # UAV has identified safe entry (blue flash) + exit (blue solid)
            # buoy positions — load them as waypoints
            safe_entry = uav_report.entry_point
            safe_exit  = uav_report.exit_point
            waypoint_queue = build_safe_passage_waypoints(safe_entry, safe_exit)
            state.current_task = SAFE_PASSAGE

        # ── CHECK FOR UUV HANDOFF (Task 2: Survey & Repair) ──────────────

        pipeline_start = READ /mission/task2/pipeline_start (non-blocking)
        IF pipeline_start is not None:
            # USV located pipeline surface marker, publishing to UUV
            PUBLISH /mission/task2/uuv_trigger → pipeline_start

        # ── CHECK FOR DOCK CONFIRMATION (Task 3: Coordinated Logistics) ──

        dock_status = READ /mission/task3/dock_status (non-blocking)
        IF dock_status.confirmed:
            # Server sends correct tin color back to USV
            tin_color = READ /mission/task3/tin_color
            PUBLISH /mission/task3/uav_delivery → {
                bay_id:    dock_status.bay_id,
                tin_color: tin_color
            }

        # ── GOAL SELECTION ────────────────────────────────────────────────

        IF state.current_goal is None AND len(waypoint_queue) > 0:
            state.current_goal = waypoint_queue.POP_FRONT()
            state.timeout_timer = now()

        # ── GOAL REACHED CHECK ────────────────────────────────────────────

        IF state.current_goal is not None:
            dist = euclidean(pose.x, pose.y,
                             state.current_goal.x, state.current_goal.y)

            IF dist < GOAL_REACHED_RADIUS:
                PUBLISH /mission/goal_reached → state.current_goal
                state.current_goal = None                 # trigger next waypoint

            # Task timeout safety
            IF now() - state.timeout_timer > TASK_TIMEOUT:
                PUBLISH /mission/task_timeout → state.current_task
                state.current_goal = None
                state.current_task = IDLE

        PUBLISH /mission/current_goal  → state.current_goal
        PUBLISH /mission/current_task  → state.current_task
```

================================================================================
## NODE 10 — ARTIFICIAL POTENTIAL FIELD (APF)
## Computes repulsive forces from obstacle grid +
## attractive force from mission goal →
## outputs target vector (surge, yaw) to low-level controller
================================================================================

```
FUNCTION apf_node():

    local_minima_counter = 0

    LOOP forever:

        grid = READ /grid/occupancy
        pose = READ /localization/pose
        goal = READ /mission/current_goal

        IF goal is None:
            PUBLISH /apf/target_vector → {surge: 0.0, yaw: 0.0}
            CONTINUE

        # ── ATTRACTIVE FORCE ──────────────────────────────────────────────

        delta_x = goal.x - pose.x
        delta_y = goal.y - pose.y
        d_goal  = sqrt(delta_x² + delta_y²)

        F_att = Vector2D(
            x = K_ATT * delta_x,
            y = K_ATT * delta_y
        )

        # ── REPULSIVE FORCES ──────────────────────────────────────────────

        F_rep = Vector2D(x=0.0, y=0.0)

        FOR each cell in grid.cells WHERE cell.occupied == true:

            # Get world position of this cell's center
            cell_x = grid.origin_x + (cell.col + 0.5) * GRID_RESOLUTION
            cell_y = grid.origin_y + (cell.row + 0.5) * GRID_RESOLUTION

            dx = pose.x - cell_x
            dy = pose.y - cell_y
            d  = sqrt(dx² + dy²)

            IF d > D0 or d < 0.001:
                CONTINUE             # outside influence radius, skip

            # Repulsive force magnitude
            magnitude = K_REP * (1.0/d - 1.0/D0) * (1.0/d²)

            # Direction: unit vector FROM obstacle TOWARD robot
            F_rep.x += magnitude * (dx / d)
            F_rep.y += magnitude * (dy / d)

        # ── TOTAL FORCE ───────────────────────────────────────────────────

        F_total = Vector2D(
            x = F_att.x + F_rep.x,
            y = F_att.y + F_rep.y
        )

        F_magnitude = sqrt(F_total.x² + F_total.y²)

        # ── LOCAL MINIMA DETECTION & ESCAPE ──────────────────────────────

        IF F_magnitude < MIN_FORCE_THRESHOLD:
            local_minima_counter += 1
        ELSE:
            local_minima_counter = 0

        IF local_minima_counter >= LOCAL_MINIMA_TICKS:
            # Inject random perturbation to escape local minimum
            random_angle = uniform(0, 2π)
            F_total.x += ESCAPE_PERTURB_MAG * cos(random_angle)
            F_total.y += ESCAPE_PERTURB_MAG * sin(random_angle)
            F_magnitude = sqrt(F_total.x² + F_total.y²)

        # ── CONVERT FORCE VECTOR TO SURGE / YAW ──────────────────────────
        # F_total is in world frame. Project onto vessel heading for surge.
        # Cross product gives yaw direction.

        # Desired heading = angle of force vector in world frame
        desired_heading = atan2(F_total.y, F_total.x)

        # Heading error (wrapped to [-π, π])
        heading_error = wrap_to_pi(desired_heading - pose.theta)

        # Surge: how much of the force aligns with current heading
        # (dot product of F_total with vessel's forward direction)
        surge = min(F_magnitude * cos(heading_error), MAX_SURGE)

        # Yaw: proportional to heading error
        yaw = clamp(heading_error, -MAX_YAW, MAX_YAW)

        PUBLISH /apf/target_vector → {surge: surge, yaw: yaw}
```

================================================================================
## NODE 11 — LOW-LEVEL CONTROLLER (Teensy)
## Translates (surge, yaw) target vector into per-thruster PWM values
## Runs PID on heading error
## Implements safety watchdog
================================================================================

```
FUNCTION low_level_controller():

    # PID state
    heading_error_integral = 0.0
    heading_error_prev     = 0.0
    last_command_time      = now()

    LOOP forever (at 50 Hz):

        # ── SAFETY WATCHDOG ───────────────────────────────────────────────

        IF now() - last_command_time > WATCHDOG_TIMEOUT:
            set_all_thrusters_pwm(STOP_PWM)
            PUBLISH /teensy/status → "WATCHDOG_TIMEOUT"
            CONTINUE

        # ── READ COMMAND ──────────────────────────────────────────────────

        target = READ /apf/target_vector (non-blocking)
        IF target is None:
            CONTINUE

        last_command_time = now()

        pose = READ /localization/pose
        dt   = 1.0 / 50.0     # 50 Hz loop

        # ── HEADING PID ───────────────────────────────────────────────────

        # target.yaw is the heading correction signal from APF
        heading_error = target.yaw

        heading_error_integral += heading_error * dt
        heading_error_derivative = (heading_error - heading_error_prev) / dt
        heading_error_prev = heading_error

        yaw_output = (KP_HEADING * heading_error
                    + KI_HEADING * heading_error_integral
                    + KD_HEADING * heading_error_derivative)

        yaw_output = clamp(yaw_output, -MAX_YAW, MAX_YAW)

        # ── SURGE SCALING ─────────────────────────────────────────────────

        surge_output = clamp(target.surge, -MAX_SURGE, MAX_SURGE)

        # ── DIFFERENTIAL THRUST ALLOCATION ───────────────────────────────
        # For a differential-drive T200 config (two thrusters, port + stbd):
        #
        #   thrust_port     = surge + yaw_correction
        #   thrust_starboard = surge - yaw_correction
        #
        # (positive yaw = turn left = port slower, starboard faster)

        thrust_port      = surge_output - yaw_output
        thrust_starboard = surge_output + yaw_output

        # Normalize if either exceeds max
        max_thrust = max(abs(thrust_port), abs(thrust_starboard))
        IF max_thrust > MAX_SURGE:
            thrust_port      /= max_thrust
            thrust_starboard /= max_thrust

        # ── CONVERT THRUST TO PWM ─────────────────────────────────────────
        # T200 expects PWM 1100–1900 μs, neutral = 1500 μs

        pwm_port      = thrust_to_pwm(thrust_port)       # linear map [-1,1]→[1100,1900]
        pwm_starboard = thrust_to_pwm(thrust_starboard)

        # ── KILL SWITCH CHECK ─────────────────────────────────────────────

        kill = READ /safety/kill_switch (non-blocking)
        IF kill.active:
            set_all_thrusters_pwm(STOP_PWM)
            PUBLISH /teensy/status → "KILLED"
            CONTINUE

        # ── OUTPUT ────────────────────────────────────────────────────────

        set_thruster_pwm(PORT,      pwm_port)
        set_thruster_pwm(STARBOARD, pwm_starboard)

        PUBLISH /teensy/telemetry → {
            pwm_port:      pwm_port,
            pwm_starboard: pwm_starboard,
            surge_cmd:     surge_output,
            yaw_cmd:       yaw_output,
            heading:       pose.theta,
            timestamp:     now()
        }

        # Telemetry feeds back into Localization Node as a
        # supplementary velocity measurement
        PUBLISH /teensy/velocity_feedback → {v: surge_output, omega: yaw_output}


FUNCTION thrust_to_pwm(thrust):
    # thrust in [-1.0, 1.0]
    # PWM range: 1100 (full reverse) to 1900 (full forward), 1500 = stop
    RETURN 1500 + thrust * 400
```

================================================================================
## NODE 12 — GCS BRIDGE + KILL SWITCH
## Relays telemetry to ground station
## Relays kill / mode commands to Teensy
================================================================================

```
FUNCTION gcs_bridge_node():

    LOOP forever:

        # ── UPLINK: vessel → GCS ──────────────────────────────────────────

        telemetry   = READ /teensy/telemetry
        pose        = READ /localization/pose
        task_status = READ /mission/current_task

        SEND_TO_GCS {
            position:   {x: pose.x, y: pose.y, heading: pose.theta},
            velocity:   {v: pose.v, omega: pose.omega},
            pwm:        telemetry,
            task:       task_status,
            timestamp:  now()
        }

        # ── DOWNLINK: GCS → vessel ────────────────────────────────────────

        gcs_cmd = RECEIVE_FROM_GCS (non-blocking)

        IF gcs_cmd.type == KILL:
            PUBLISH /safety/kill_switch → {active: true}

        ELSE IF gcs_cmd.type == SET_MODE:
            PUBLISH /safety/mode → gcs_cmd.mode   # AUTO / MANUAL / KILLED

        ELSE IF gcs_cmd.type == OVERRIDE_GOAL:
            PUBLISH /mission/current_goal → gcs_cmd.goal

        SLEEP 0.1
```

================================================================================
## MAIN — ROS2 NODE LAUNCH
## Starts all nodes as parallel ROS2 processes
================================================================================

```
FUNCTION main():

    ros2_init()

    # Launch all nodes as concurrent ROS2 nodes
    SPAWN lidar_api_node()
    SPAWN camera_api_node()
    SPAWN gps_api_node()
    SPAWN imu_api_node()

    SPAWN localization_node()       # EKF: GPS + IMU + COMPASS → pose
    SPAWN sensor_fusion_node()      # voxel + ICP + projection → labeled cloud
    SPAWN coordinate_transform_node()  # body frame → world frame

    SPAWN occupancy_grid_node()     # world-frame 2D obstacle map
    SPAWN mission_manager_node()    # state machine, goal publisher
    SPAWN apf_node()                # polarity field → surge/yaw vector

    SPAWN low_level_controller()    # Teensy: PID + thrust allocation + PWM
    SPAWN gcs_bridge_node()         # telemetry uplink + kill switch

    ros2_spin()   # block until shutdown
```

================================================================================
## ROS2 TOPIC MAP SUMMARY
================================================================================

```
TOPIC                           TYPE            PUBLISHER           SUBSCRIBER(S)
─────────────────────────────────────────────────────────────────────────────────
/lidar/raw_cloud                PointCloud      lidar_api           sensor_fusion
/camera/detections              Detection[]     camera_api          sensor_fusion
/camera/frame                   Image           camera_api          sensor_fusion
/gps/position                   Point2D         gps_api             localization
/imu/data                       IMUData         imu_api             localization
/localization/pose              Pose            localization        coord_transform
                                                                    apf
                                                                    mission_manager
                                                                    occupancy_grid
                                                                    gcs_bridge
/localization/covariance        Matrix5x5       localization        (debug/GCS)
/fusion/labeled_cloud_body      PointCloud      sensor_fusion       coord_transform
/fusion/labeled_cloud_world     PointCloud      coord_transform     occupancy_grid
/grid/occupancy                 OccupancyGrid   occupancy_grid      apf
/mission/current_goal           Point2D         mission_manager     apf
                                                                    occupancy_grid
/mission/current_task           TaskEnum        mission_manager     gcs_bridge
/mission/goal_reached           Point2D         mission_manager     (triggers next)
/mission/task1/report           BuoyReport      UAV (external)      mission_manager
/mission/task2/uuv_trigger      Point2D         mission_manager     UUV (external)
/mission/task3/uav_delivery     DeliveryCmd     mission_manager     UAV (external)
/apf/target_vector              Vector2D        apf                 low_level_ctrl
/teensy/telemetry               TelemetryMsg    low_level_ctrl      gcs_bridge
/teensy/velocity_feedback       VelocityMsg     low_level_ctrl      localization
/safety/kill_switch             BoolMsg         gcs_bridge          low_level_ctrl
/safety/mode                    ModeEnum        gcs_bridge          low_level_ctrl
```