from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def package_launch(package: str, filename: str, arguments=None, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(package), "launch", filename])
        ),
        launch_arguments=(arguments or {}).items(),
        condition=condition,
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("model_config", default_value=""),
            DeclareLaunchArgument("checkpoint", default_value=""),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument("device_id", default_value="0"),
            DeclareLaunchArgument("start_pca9685", default_value="false"),
            DeclareLaunchArgument("start_recording", default_value="true"),
            DeclareLaunchArgument("capture_root", default_value="data/autonomy_bags"),
            DeclareLaunchArgument("record_mask", default_value="false"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
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
                "adom_planning",
                "semantic20_local_planning.launch.py",
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ),
            package_launch(
                "adom_control",
                "gamepad_control.launch.py",
                {
                    "device_id": LaunchConfiguration("device_id"),
                    "start_pca9685": LaunchConfiguration("start_pca9685"),
                    "start_data_recorder": "false",
                },
            ),
            package_launch(
                "adom_logging",
                "autonomy_logging.launch.py",
                {
                    "capture_root": LaunchConfiguration("capture_root"),
                    "record_mask": LaunchConfiguration("record_mask"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                },
                condition=IfCondition(LaunchConfiguration("start_recording")),
            ),
        ]
    )
