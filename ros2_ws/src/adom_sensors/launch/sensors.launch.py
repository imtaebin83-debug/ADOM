from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_zed = LaunchConfiguration("use_zed")
    use_gnss = LaunchConfiguration("use_gnss")
    zed_config = PathJoinSubstitution([FindPackageShare("adom_sensors"), "config", "zed2i.yaml"])
    gnss_config = PathJoinSubstitution([FindPackageShare("adom_sensors"), "config", "rtk_gnss.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("use_zed", default_value="true"),
        DeclareLaunchArgument("use_gnss", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare("zed_wrapper"), "launch", "zed_camera.launch.py"])
            ),
            launch_arguments={
                "camera_model": "zed2i",
                "ros_params_override_path": zed_config,
            }.items(),
            condition=IfCondition(use_zed),
        ),
        Node(
            package="nmea_navsat_driver",
            executable="nmea_serial_driver",
            name="nmea_serial_driver",
            parameters=[gnss_config],
            remappings=[("fix", "/fix")],
            output="screen",
            condition=IfCondition(use_gnss),
        ),
    ])
