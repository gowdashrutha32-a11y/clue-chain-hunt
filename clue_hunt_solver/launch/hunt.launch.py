"""
Starts YOUR leader and follower nodes. This is the single command used during evaluation
(after the simulation + Nav2 are up):

  ros2 launch clue_hunt_solver hunt.launch.py

Add parameters / extra nodes here as you need them.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='clue_hunt_solver', executable='hunt_node', output='screen',
             parameters=[{'use_sim_time': True}]),
        Node(package='clue_hunt_solver', executable='follower_node', output='screen',
             parameters=[{'use_sim_time': True}]),
    ])
