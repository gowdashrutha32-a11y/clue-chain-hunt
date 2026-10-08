"""
Mapping: simulation + slam_toolbox + RViz. Drive with teleop, then save the map.

  ros2 launch clue_hunt_navigation mapping.launch.py
  ros2 run teleop_twist_keyboard teleop_twist_keyboard          # other terminal
  ros2 run nav2_map_server map_saver_cli -f ~/hunt_ws/maps/arena  # when done
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

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, 'launch', 'sim.launch.py')),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'world_pkg': LaunchConfiguration('world_pkg'),
            'gui': LaunchConfiguration('gui'),
            'follower': LaunchConfiguration('follower'),
        }.items(),
    )

    slam = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[os.path.join(nav_share, 'config', 'slam_params.yaml'), {'use_sim_time': True}],
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
        DeclareLaunchArgument('follower', default_value='false', description='Spawn the follower too'),
        sim,
        TimerAction(period=5.0, actions=[slam, rviz]),
    ])
