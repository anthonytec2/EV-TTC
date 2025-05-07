import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

import launch
from launch.actions import DeclareLaunchArgument as LaunchArg
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration as LaunchConfig


def generate_launch_description():
    """Generate launch description with multiple components."""
    cam_name = "event_camera"
    pkg_name = "metavision_driver"
    share_dir = get_package_share_directory(pkg_name)
    trigger_config = os.path.join(share_dir, "config", "trigger_pins.yaml")

    container = ComposableNodeContainer(
        name="events",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="metavision_driver",
                plugin="metavision_driver::DriverROS2",
                name="driver",
                parameters=[
                    trigger_config,  # loads the whole file
                    {
                        "use_multithreading": False,
                        "statistics_print_interval": 2.0,
                        # 'bias_file': bias_config,
                        "camerainfo_url": "",
                        "frame_id": "",
                        "serial": "",
                        "erc_mode": "enabled",
                        "erc_rate": 70000000,
                        "roi": [280, 0, 720, 720],
                        # valid: 'external', 'loopback', 'disabled'
                        "trigger_in_mode": "external",
                        "mipi_frame_period": 2000,
                        # valid: 'enabled', 'disabled'
                        "trigger_out_mode": "enabled",
                        "trigger_out_period": 100000,  # in usec
                        "trigger_duty_cycle": 0.5,  # fraction high/low
                        "event_message_time_threshold": 1.0e-3,
                    },
                ],
                remappings=[("~/events", "event_camera/events")],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            ComposableNode(
                package="rss",
                plugin="rss::Exp",
                name="exp_node",
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
        output="both",
    )

    return launch.LaunchDescription([container])
