from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    model = PathJoinSubstitution(
        [FindPackageShare("adom_description"), "urdf", "adom_vehicle.urdf.xacro"]
    )
    robot_description = {"robot_description": Command([FindExecutable(name="xacro"), " ", model])}
    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[robot_description],
            output="screen",
        )
    ])
