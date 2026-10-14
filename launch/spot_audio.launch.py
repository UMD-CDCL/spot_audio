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
            # Respawn so a crash or a fatal start error self-heals. The Pi keeps
            # streaming regardless, so a respawn just re-attaches to the stream.
            respawn=True,
            respawn_delay=2.0,
        ),
        # audio_classification_node (AST + faster-whisper) was replaced by
        # spot_assessment_node, which scores the assessment's audio with Gemma,
        # transcribes the casualty's replies with Whisper, and grades them. It still
        # needs microphone_node's raw_audio and speaker_node's speaker/voice (the
        # robot's own questions), so both stay. speaker_node's stop_listening client
        # now has no server; its call is fire-and-forget, so that is harmless.
        Node(
            package='spot_audio',
            namespace=spot_name,
            executable='speaker_node.py',
            name='speaker_node',
            output='log'
        )
    ])
