# Clue Chain Hunt

ROS 2 autonomous multi-robot clue-solving system developed for the Clue Chain Hunt problem.

## System Overview

The system consists of two autonomous robots:

### Leader / Hunter

The leader robot:

1. Receives camera images.
2. Detects ArUco markers.
3. Detects and decodes QR clues.
4. Validates clue IDs and cryptographic tokens.
5. Resolves clue commands such as:
   - `GOTO`
   - `PILLAR`
   - `BETWEEN`
   - `REL`
   - `TREASURE REL`
6. Uses Nav2 for autonomous navigation.
7. Publishes detected board locations.
8. Publishes the final treasure location.

### Follower

The follower robot:

1. Uses its camera to detect ArUco ID 49 on the leader.
2. Estimates the relative pose of the leader.
3. Uses its own wheel odometry.
4. Maintains a safe following distance.
5. Publishes velocity commands on `/follower/cmd_vel`.

The follower does not use the leader's pose, LiDAR, map, or ground-truth Gazebo state.

---

## Repository Structure

```text
clue-chain-hunt/
│
├── clue_hunt_solver/
│   ├── package.xml
│   ├── setup.py
│   ├── setup.cfg
│   ├── resource/
│   │   └── clue_hunt_solver
│   │
│   ├── clue_hunt_solver/
│   │   ├── __init__.py
│   │   ├── hunt_node.py
│   │   └── follower_node.py
│   │
│   └── launch/
│       └── hunt.launch.py
│
├── clue_hunt_navigation/
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── config/
│   │   ├── nav2_params.yaml
│   │   └── slam_params.yaml
│   ├── launch/
│   │   ├── mapping.launch.py
│   │   └── navigation.launch.py
│   └── rviz/
│       └── hunt.rviz
│
├── maps/
│   ├── arena.yaml
│   └── arena.pgm
│
└── README.md
