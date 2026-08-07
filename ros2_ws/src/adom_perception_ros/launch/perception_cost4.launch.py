from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("adom_perception_ros"), "config", "perception.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_config),
            DeclareLaunchArgument("model_config", default_value=""),
            DeclareLaunchArgument("checkpoint", default_value=""),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            Node(
                package="adom_perception_ros",
                executable="adom_cost4_perception_node",
                name="adom_perception",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "config_path": LaunchConfiguration("model_config"),
                        "checkpoint_path": LaunchConfiguration("checkpoint"),
                        "device": LaunchConfiguration("device"),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    },
                ],
                output="screen",
            ),
        ]
    )
