from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mask_topic",
                default_value="/adom/perception/semantic20_mask_evidence",
            ),
            DeclareLaunchArgument(
                "color_topic",
                default_value="/adom/perception/semantic20_mask_color",
            ),
            Node(
                package="adom_perception_ros",
                executable="semantic20_colorizer_node",
                name="semantic20_colorizer",
                parameters=[
                    {
                        "mask_topic": LaunchConfiguration("mask_topic"),
                        "color_topic": LaunchConfiguration("color_topic"),
                    }
                ],
                output="screen",
            ),
        ]
    )
