from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("adom_planning")
    executor_config = PathJoinSubstitution([share, "config", "waypoint_executor.yaml"])
    nav2_launch = PathJoinSubstitution([share, "launch", "planning.launch.py"])
    return LaunchDescription([
        DeclareLaunchArgument(
            "waypoint_file",
            description="Absolute path to a WGS84 waypoint YAML file",
        ),
        DeclareLaunchArgument("start_nav2", default_value="true"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={"use_sim_time": LaunchConfiguration("use_sim_time")}.items(),
            condition=IfCondition(LaunchConfiguration("start_nav2")),
        ),
        Node(
            package="adom_planning",
            executable="rtk_waypoint_executor",
            name="rtk_waypoint_executor",
            parameters=[
                executor_config,
                {
                    "waypoint_file": LaunchConfiguration("waypoint_file"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                },
            ],
            output="screen",
        ),
    ])
