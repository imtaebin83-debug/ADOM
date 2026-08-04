from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("adom_localization")
    local = PathJoinSubstitution([share, "config", "ekf_local.yaml"])
    global_ = PathJoinSubstitution([share, "config", "ekf_global.yaml"])
    navsat = PathJoinSubstitution([share, "config", "navsat.yaml"])
    return LaunchDescription([
        Node(
            package="robot_localization", executable="ekf_node", name="ekf_local_filter",
            parameters=[local], remappings=[("odometry/filtered", "/odometry/local")], output="screen"
        ),
        Node(
            package="robot_localization", executable="ekf_node", name="ekf_global_filter",
            parameters=[global_], remappings=[("odometry/filtered", "/odometry/global")], output="screen"
        ),
        Node(
            package="robot_localization", executable="navsat_transform_node", name="navsat_transform",
            parameters=[navsat],
            remappings=[
                ("imu", "/zed/zed_node/imu/data"),
                ("gps/fix", "/fix"),
                ("odometry/filtered", "/odometry/global"),
                ("odometry/gps", "/odometry/gps"),
                ("gps/filtered", "/gps/filtered"),
            ],
            output="screen",
        ),
    ])
