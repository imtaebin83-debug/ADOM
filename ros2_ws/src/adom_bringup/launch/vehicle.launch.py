from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def include(package, launch_file, enabled):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(package), "launch", launch_file])
        ),
        condition=IfCondition(LaunchConfiguration(enabled)),
    )


def generate_launch_description():
    flags = [
        ("start_description", "true"),
        ("start_sensors", "true"),
        ("start_localization", "true"),
        ("start_planning", "false"),
        ("start_control", "true"),
    ]
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=value) for name, value in flags]
        + [
            include("adom_description", "description.launch.py", "start_description"),
            include("adom_sensors", "sensors.launch.py", "start_sensors"),
            include("adom_localization", "localization.launch.py", "start_localization"),
            include("adom_planning", "planning.launch.py", "start_planning"),
            include("adom_control", "control.launch.py", "start_control"),
        ]
    )
