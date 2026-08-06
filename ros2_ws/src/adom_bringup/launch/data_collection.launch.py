from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def include(package, launch_file, arguments):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(package), "launch", launch_file])
        ),
        launch_arguments=arguments.items(),
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_zed", default_value="true"),
            DeclareLaunchArgument("use_gnss", default_value="true"),
            DeclareLaunchArgument("device_id", default_value="0"),
            DeclareLaunchArgument("start_pca9685", default_value="true"),
            DeclareLaunchArgument("capture_root", default_value=""),
            include(
                "adom_sensors",
                "sensors.launch.py",
                {
                    "use_zed": LaunchConfiguration("use_zed"),
                    "use_gnss": LaunchConfiguration("use_gnss"),
                },
            ),
            include(
                "adom_control",
                "gamepad_control.launch.py",
                {
                    "device_id": LaunchConfiguration("device_id"),
                    "start_pca9685": LaunchConfiguration("start_pca9685"),
                    "start_data_recorder": "true",
                    "capture_root": LaunchConfiguration("capture_root"),
                },
            ),
        ]
    )
