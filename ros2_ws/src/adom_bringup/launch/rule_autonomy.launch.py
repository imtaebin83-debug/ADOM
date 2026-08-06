from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def package_launch(package: str, filename: str, arguments=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(package), "launch", filename])
        ),
        launch_arguments=(arguments or {}).items(),
    )


def generate_launch_description():
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("adom_bringup"), "config", "rule_autonomy.rviz"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("model_config", default_value=""),
            DeclareLaunchArgument("checkpoint", default_value=""),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("start_rviz", default_value="true"),
            package_launch(
                "adom_perception_ros",
                "perception.launch.py",
                {
                    "model_config": LaunchConfiguration("model_config"),
                    "checkpoint": LaunchConfiguration("checkpoint"),
                    "device": LaunchConfiguration("device"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                },
            ),
            package_launch(
                "adom_costmap_ros",
                "semantic_costmap.launch.py",
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ),
            package_launch(
                "adom_planning",
                "rule_planning.launch.py",
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="adom_rule_rviz",
                arguments=["-d", rviz_config],
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_rviz")),
            ),
        ]
    )
