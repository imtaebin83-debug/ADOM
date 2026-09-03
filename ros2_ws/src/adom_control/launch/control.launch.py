from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([FindPackageShare("adom_control"), "config", "vehicle.yaml"])
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            # pca9685_control initializes real I2C hardware and drives PWM. The
            # default stays true so existing bringup behaviour is unchanged; pass
            # start_pca9685:=false for a dry run with no actuator output.
            DeclareLaunchArgument("start_pca9685", default_value="true"),
            Node(
                package="adom_control",
                executable="pca9685_control",
                name="pca9685_control",
                parameters=[LaunchConfiguration("config")],
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_pca9685")),
            ),
        ]
    )
