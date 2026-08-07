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
            DeclareLaunchArgument("start_data_recorder", default_value="true"),
            DeclareLaunchArgument("capture_root", default_value="data/captures"),
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
                parameters=[LaunchConfiguration("config")],
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_pca9685")),
            ),
            Node(
                package="adom_control",
                executable="data_recorder",
                name="data_recorder",
                parameters=[
                    LaunchConfiguration("config"),
                    {"capture_root": LaunchConfiguration("capture_root")},
                ],
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_data_recorder")),
            ),
        ]
    )
