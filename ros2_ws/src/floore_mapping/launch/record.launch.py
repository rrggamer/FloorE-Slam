"""Bring up the FloorE Gazebo sim and start recording synchronized odom+scan
data to a CSV file. Drive the robot around in a separate terminal with:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard
Stop recording with Ctrl+C once you've covered the area you want mapped, then
run floore_mapping/build_ocgm.py on the resulting CSV.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    gazebo_share = get_package_share_directory("floore_gazebo")

    out_file_arg = DeclareLaunchArgument("out_file", default_value="data.csv")

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

    return LaunchDescription([out_file_arg, gazebo, recorder])
