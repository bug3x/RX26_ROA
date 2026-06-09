# RX26 Reactive Obstacle Avoidance

================================================================================
## CONSTANTS & CONFIGURATION==
==============================================================================

```
[ENTER]
# --- APF ---
K_ATT = VAL1 # attractive gain
K_REP = VAL2 # repulsive gain
INF_RAD      =        # influence radius
MIN_FORCE_THRESHOLD = # detection for local minima
LOCAL_MINIMA_TICKS  = # confirm local minima for random movement
```

================================================================================
## DATA STRUCTURES
================================================================================

```
[ENTER]
```

================================================================================
## NODE 1 — LIDAR API
## Produces raw 360° point cloud each scan cycle (~10 Hz)
================================================================================

```
[ENTER]
```

================================================================================
## NODE 2 — CAMERA API
## Produces YOLO bounding boxes + class labels at ~30 Hz
================================================================================

```
[ENTER]
```

================================================================================
## NODE 3 — GPS API
## Publishes raw lat/lon at ~5 Hz
================================================================================

```
[ENTER]
```

================================================================================
## NODE 4 — IMU API
## Publishes accelerometer + gyroscope + compass at ~100 Hz
================================================================================

```
[ENTER]
```

================================================================================
## NODE 5 — LOCALIZATION NODE (EKF)
## Fuses GPS + IMU + COMPASS → world pose estimate
## Extended Kalman Filter implimentation
================================================================================

```
[ENTER]
```

================================================================================
## NODE 6 — SENSOR FUSION
## Stage A: Voxel filter + outlier removal on LiDAR
## Stage B: ICP-based projection (pixel → point cloud mapping)
## only geometric processing, no kalman filter
================================================================================

```
[ENTER]
```

================================================================================
## NODE 7 — COORDINATE TRANSFORM NODE
## Rotates body-frame obstacle cloud into world frame using EKF pose
================================================================================

```
[ENTER]
```

================================================================================
## NODE 8 — OCCUPANCY GRID
## Maintains a 2D world-frame map of obstacle cells
## Includes cell aging / decay to remove stale data
================================================================================

```
[ENTER]
```

================================================================================
## NODE 9 — MISSION MANAGER (State Machine)
## Determines current goal, handles task sequencing,
## subscribes to UAV reports for RobotX 2026 cross-domain handoff
================================================================================

```
[ENTER]
```

================================================================================
## NODE 10 — ARTIFICIAL POTENTIAL FIELD (APF)
## Computes repulsive forces from obstacle grid +
## attractive force from mission goal →
## outputs target vector (surge, yaw) to low-level controller
================================================================================

```
[ENTER]
FUNCTION apf_node():
    local_minima_counter = 0

    LOOP forever:

        grid = READ /grid/occupancy
        pose = READ /localization/pose
        target = READ /mission/targets

        IF goal is NONE:
            PUBLISH /apf/target_vector -> {surge: 0.0, yaw: 0.0}
            CONTINUE
        
        # ATTRACTIVE FORCE
        delta_x = goal.x - pose.x
        delta_y = goal.y - pose.y
        d_goal = sqrt(delta_x^2 + delta_y^2)

        F_att = Vector2D(
            x = K_ATT * delta_x,
            y = K_ATT * delta_y
        )

        # REPULSIVE FORCE
        F_rep = Vector2D(x = 0.0, y = 0.0)
        FOR each cell in grid.cells WHERE the cell.occupied == true:

            # get the world position of cell's center
            cell_x = grid.origin.x + (cell.col)
            cell_y = grid.origin.y + (cell.row)

            dx = pose.x - cell_x
            dy = pose.y - cell_y
            d = sqrt(dx^2 + dy^2)

            # skip if object is out of distance, out of influence radius
            IF d > INF_RAD OR d < 0.001:
                CONTINUE

            # repulsive force magnitude
            magnitude = K_REP * (1.0/d - 1.0/INF_RAD) * (1.0/d²) # how magnitude?
            
            # Direction: unit vector FROM obstacle TOWARD robot
            F_rep.x += magnitude * (dx / d)
            F_rep.y += magnitude * (dy / d)
        
        # TOTAL FORCE
        F_total = Vector2D(
            x = F_att.x + F_rep.x,
            y = F_att.y + F_rep.y
        )
        magnitude = sqrt(F_total.x^2 + F_total.y^2)

        # MINIMA
        [NOT YET IMPLEMENTED]
        # General Logic: set params MIN_FORCE_THRESHOLD and LOCAL_MINIMA_TICKS
        # On conditions based on the ticks ex. magnitude < MIN_FORCE_THRESHOLD 
        # LOCAL_MINIMA_TICKS met -> random movement out of the minima

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

        PUBLISH /apt/target_vector -> Vector2D(surge: surge, yaw: yaw)
```

================================================================================
## NODE 11 — LOW-LEVEL CONTROLLER (Teensy)
## Translates (surge, yaw) target vector into per-thruster PWM values
## Runs PID on heading error
## Implements safety watchdog
================================================================================

```
[ENTER]
```

================================================================================
## NODE 12 — GCS BRIDGE + KILL SWITCH
## Relays telemetry to ground station
## Relays kill / mode commands to Teensy
================================================================================

```
[ENTER]
```

================================================================================
## MAIN — ROS2 NODE LAUNCH
## Starts all nodes as parallel ROS2 processes
================================================================================

```
[ENTER]
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