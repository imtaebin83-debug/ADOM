from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("adom_logging"), "config", "autonomy_logging.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_config),
            DeclareLaunchArgument("capture_root", default_value="data/autonomy_bags"),
            DeclareLaunchArgument("record_mask", default_value="false"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            Node(
                package="adom_logging",
                executable="gps_track_logger",
                name="gps_track_logger",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
                output="screen",
            ),
            Node(
                package="adom_control",
                executable="data_recorder",
                name="autonomy_data_recorder",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "capture_root": LaunchConfiguration("capture_root"),
                        "record_mask": ParameterValue(
                            LaunchConfiguration("record_mask"), value_type=bool
                        ),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    },
                ],
                output="screen",
            ),
        ]
    )
