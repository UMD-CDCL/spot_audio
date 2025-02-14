from launch import LaunchDescription
from launch_ros.actions import LifecycleNode

def generate_launch_description():
    return LaunchDescription([
        LifecycleNode(
            package='spot_audio',
            namespace='',
            executable='microphone_lifecycle_node.py',
            name='microphone_lifecycle_node',
            output='screen'
        )
    ])