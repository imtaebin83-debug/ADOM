from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([FindPackageShare("adom_control"), "config", "vehicle.yaml"])
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            Node(
                package="adom_control",
                executable="pca9685_control",
                name="pca9685_control",
                parameters=[LaunchConfiguration("config")],
                output="screen",
            ),
        ]
    )
