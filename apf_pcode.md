=============================================================================
## EXTERNAL PARAMETER
=============================================================================
```
K ATTRACTIVE FORCE GAIN :   [-]
K REPULSIVE FORCE GAIN :   [-]
RADIUS OF INFLUENCE :   [m]

LOCAL MINIMA TICKERS :   [-]
# ticks to enforce minima escape function
MIN FORCE THRESHOLD :   [-]
# minimum force to determine local minima and start local minima ticker
RANDOM PERTUBATION :   []
# magnitude for random movement to escape the local minima

MAX SURGE :   [m/s]
MAX YAW :   [rad/s]

TARGET TOLERANCE : [m]
# arrival radius to stop local minima near the target

FIXED FRAME : "map"
# all topics transformed to this frame
LOOP HZ : 10 [Hz]
# control loop frequency
```
==============================================================================
## ARTIFICIAL POTENTIAL FIELD
## CONSIDER: APF does not take in kinematic restraints (e.c. vehicle acceleration limits, braking)
## CONSIDER: Hybrid usage with Dynamic Window Approach (DWA),
## which accounts for kinematic restraints to reduce drift and ensure smoother vehicle motion
==============================================================================
'''
FUNCTION apf_node():
    local_minima_ticker = 0

    LOOP FOREVER at LOOP HZ:
        # get subscriber inputs
        grid = READ occupancy topic
        pose = READ pose topic
        target = READ current target topic

        IF target is None:
            PUBLISH target vector = {surge = 0.0, yaw = 0.0}

        # ATTRACTIVE FORCE
        # get the distance to the goal and x and y displacement between the goal and pose, output a 2D vector containing K ATT.x and K ATT.y

        # REPULSIVE FORCE
        # distribute the grid into cells, traverse through occupied cells to determine x and y distance, output a 2D vector containing K REP.x and K REP.y
        # check if radius of influence is met

        # TOTAL FORCE
        # get the magnitude of attractive force and repulsive force in both x and y positions to optimize the vector path

        # LOCAL MINIMA ESCAPE

        # FORCE VECTOR CONVERSION TO VELOCITY VECTORS (surge, yaw)
