from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def package_launch(package: str, filename: str, arguments: dict):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(package), "launch", filename])
        ),
        launch_arguments=arguments.items(),
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            package_launch(
                "adom_costmap_ros",
                "semantic20_costmap.launch.py",
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ),
            package_launch(
                "adom_planning",
                "rule_planning.launch.py",
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ),
            package_launch(
                "adom_control",
                "local_path_control.launch.py",
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ),
        ]
    )
