"""
Navigation: simulation + Nav2 (AMCL + saved map) + RViz.

  ros2 launch clue_hunt_navigation navigation.launch.py map:=$HOME/hunt_ws/maps/arena.yaml
  ros2 launch clue_hunt_navigation navigation.launch.py sim:=false ...   # Nav2 only (sim already running)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    nav_share = get_package_share_directory('clue_hunt_navigation')
    gz_share = get_package_share_directory('clue_hunt_gazebo')
    bringup_share = get_package_share_directory('nav2_bringup')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, 'launch', 'sim.launch.py')),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'world_pkg': LaunchConfiguration('world_pkg'),
            'gui': LaunchConfiguration('gui'),
            'follower': LaunchConfiguration('follower'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('sim')),
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup_share, 'launch', 'bringup_launch.py')),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'params_file': LaunchConfiguration('params_file'),
            'use_sim_time': 'true',
            'autostart': 'true',
            'slam': 'False',
        }.items(),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(nav_share, 'rviz', 'hunt.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='log',
    )

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='practice'),
        DeclareLaunchArgument('world_pkg', default_value='clue_hunt_gazebo'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('follower', default_value='true', description='Spawn the follower too'),
        DeclareLaunchArgument('sim', default_value='true', description='Also start Gazebo + robot'),
        DeclareLaunchArgument('map', default_value=os.path.expanduser('~/hunt_ws/maps/arena.yaml'),
                              description='Full path to the map yaml'),
        DeclareLaunchArgument('params_file', default_value=os.path.join(nav_share, 'config', 'nav2_params.yaml')),
        sim,
        TimerAction(period=6.0, actions=[nav2, rviz]),
    ])
