from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("adom_control"), "config", "vehicle.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("device_id", default_value="0"),
            DeclareLaunchArgument("start_pca9685", default_value="true"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            Node(
                package="joy",
                executable="joy_node",
                name="joy_node",
                parameters=[
                    {
                        "device_id": ParameterValue(
                            LaunchConfiguration("device_id"), value_type=int
                        ),
                        "deadzone": 0.05,
                        "autorepeat_rate": 30.0,
                    }
                ],
                output="screen",
            ),
            Node(
                package="adom_control",
                executable="gamepad_control",
                name="gamepad_control",
                parameters=[LaunchConfiguration("config")],
                output="screen",
            ),
            Node(
                package="adom_control",
                executable="pca9685_control",
                name="pca9685_control",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "dry_run": ParameterValue(
                            LaunchConfiguration("dry_run"), value_type=bool
                        )
                    },
                ],
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_pca9685")),
            ),
        ]
    )
