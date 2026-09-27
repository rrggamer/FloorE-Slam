"""Bring up Gazebo Sim with the FloorE world, spawn the robot, and bridge
ROS2 <-> Gazebo topics (cmd_vel, odom, tf, scan, joint_states, clock)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    UnsetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# If this is launched from a snap-packaged terminal (e.g. VSCode's built-in
# terminal, which ships as a snap), these env vars point at the terminal
# app's own bundled GTK/libc, not the system ones. gz sim's GUI inherits
# them, dlopens a GTK module from inside that snap, and crashes with
# "libpthread.so.0: undefined symbol: __libc_pthread_init, version
# GLIBC_PRIVATE" -- a glibc-version clash between the snap's bundled libc
# and the host's. Server-only mode never hits this (no GUI/GTK loading), so
# it's only visible with gz_args="-r ...". Unsetting them here has no
# effect outside a snap-confined shell.
_SNAP_GUI_ENV_VARS = [
    "SNAP", "SNAP_LIBRARY_PATH", "SNAP_NAME", "SNAP_REVISION", "SNAP_ARCH",
    "SNAP_INSTANCE_NAME", "SNAP_UID", "SNAP_EUID", "SNAP_CONTEXT", "SNAP_COOKIE",
    "SNAP_REAL_HOME", "SNAP_USER_COMMON", "SNAP_USER_DATA", "SNAP_DATA",
    "SNAP_COMMON", "SNAP_VERSION", "SNAP_LAUNCHER_ARCH_TRIPLET",
    "GTK_PATH", "GTK_EXE_PREFIX", "GDK_PIXBUF_MODULE_FILE", "GDK_PIXBUF_MODULEDIR",
    "GIO_MODULE_DIR", "GTK_IM_MODULE_FILE", "LOCPATH",
]


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

    unset_snap_gui_vars = [
        UnsetEnvironmentVariable(name) for name in _SNAP_GUI_ENV_VARS if name in os.environ
    ]

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

    # z=2.0: the robot free-falls onto the floor at spawn, which has been
    # observed to occasionally land it hard enough to flip over (confirmed
    # via /ground_truth pose logging in one run). Dropping to z=0.06 (the
    # robot's actual wheel clearance) avoided that in testing, but visibly
    # clips the wheels into the floor mesh instead, so this is being kept at
    # 2.0 -- if a fall ever lands it wrong, just relaunch, or nudge it
    # upright with `gz service` before driving.
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

    # lidar_joint is a fixed joint, so gz-sim/dartsim welds lidar_link into
    # base_link at load time (confirmed via `gz model -m floore -l`: the
    # lidar sensor shows up directly under base_link, no separate
    # lidar_link entity exists at runtime). That changes the LaserScan
    # message's frame_id to the scoped "floore/base_link/lidar" instead of
    # "lidar_link", which robot_state_publisher's TF tree has no entry for
    # -- RViz then reports the frame as missing. Bridging the two names
    # with the sensor's own known offset (from floore.urdf.xacro's
    # lidar_x/y/z) fixes it without fighting the welding behavior.
    lidar_frame_bridge = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0.033635", "--y", "0.0", "--z", "0.75618",
            "--frame-id", "base_link", "--child-frame-id", "floore/base_link/lidar",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            set_resource_path,
            *unset_snap_gui_vars,
            gz_sim,
            robot_state_publisher,
            spawn_robot,
            bridge,
            lidar_frame_bridge,
        ]
    )
