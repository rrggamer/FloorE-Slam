"""Fully autonomous data collection: brings up the FloorE Gazebo sim, an
obstacle-avoidance wander node (explore.py) driving cmd_vel, and the
scan+ground-truth recorder (record_data.py) -- one command instead of
driving manually with teleop_twist_keyboard. Stop with Ctrl+C once
coverage looks like enough for build_ocgm.py, or set run_time to have the
explorer (only) stop itself after N seconds.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    gazebo_share = get_package_share_directory("floore_gazebo")

    out_file_arg = DeclareLaunchArgument("out_file", default_value="data.csv")
    run_time_arg = DeclareLaunchArgument(
        "run_time", default_value="0.0", description="seconds; <=0 means run until Ctrl+C"
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, "launch", "gazebo.launch.py")
        )
    )

    recorder = Node(
        package="floore_mapping",
        executable="record_data.py",
        output="screen",
        parameters=[{"out_file": LaunchConfiguration("out_file")}],
    )

    explorer = Node(
        package="floore_mapping",
        executable="explore.py",
        output="screen",
        parameters=[{"run_time": LaunchConfiguration("run_time")}],
    )

    # When run_time > 0, explore.py exits on its own once the timer fires
    # (see explore.py's _time_up). Tie that exit to shutting down the whole
    # launch (Gazebo, bridge, recorder included) so a timed run is truly
    # one command with nothing left running after it. With the default
    # run_time=0, explore.py only exits on Ctrl+C, which already takes the
    # whole process group down, so this handler is a no-op in that case.
    stop_all_when_explorer_exits = RegisterEventHandler(
        OnProcessExit(target_action=explorer, on_exit=[EmitEvent(event=Shutdown())])
    )

    return LaunchDescription(
        [out_file_arg, run_time_arg, gazebo, recorder, explorer, stop_all_when_explorer_exits]
    )
