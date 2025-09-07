from launch import LaunchDescription
from launch_ros.actions import Node
import os

# get the home directory
home = os.environ['HOME']

# get Spot's name from a global variable
spot_name = os.environ['SPOT_NAME']

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='spot_audio',
            namespace=spot_name,
            executable='microphone_node.py',
            name='microphone_node',
            output='log',
        ),
        Node(
            package='spot_audio',
            namespace=spot_name,
            executable='audio_classification_node.py',
            name='audio_classification_node',
            output='log',
        ),
        Node(
            package='spot_audio',
            namespace=spot_name,
            executable='speaker_node.py',
            name='speaker_node',
            output='log'
        )
    ])
