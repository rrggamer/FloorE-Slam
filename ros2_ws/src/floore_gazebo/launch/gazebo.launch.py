"""Bring up Gazebo Sim with the FloorE world, spawn the robot, and bridge
ROS2 <-> Gazebo topics (cmd_vel, odom, tf, scan, joint_states, clock)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    description_share = get_package_share_directory("floore_description")
    gazebo_share = get_package_share_directory("floore_gazebo")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")

    xacro_path = os.path.join(description_share, "urdf", "floore.urdf.xacro")
    world_path = os.path.join(gazebo_share, "worlds", "floore_world.sdf")
    bridge_config_path = os.path.join(gazebo_share, "config", "bridge.yaml")

    # Resolve model://floore_description/... URIs in floore_world.sdf: the
    # first URI segment (floore_description) is looked up as a subdirectory
    # of each path listed here, so pointing at the share/ root is enough --
    # no per-package entry or model.config needed.
    share_root = os.path.dirname(description_share)
    existing_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    resource_path = (
        f"{share_root}:{existing_resource_path}" if existing_resource_path else share_root
    )

    set_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH", value=resource_path
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": f"-r {world_path}"}.items(),
    )

    robot_description = ParameterValue(
        Command(["xacro ", xacro_path]), value_type=str
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-topic", "robot_description", "-name", "floore", "-x", "2.5", "-y", "2.5", "-z", "2.0"],
        output="screen",
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{"config_file": bridge_config_path, "use_sim_time": True}],
        output="screen",
    )

    return LaunchDescription(
        [
            set_resource_path,
            gz_sim,
            robot_state_publisher,
            spawn_robot,
            bridge,
        ]
    )
