#!/usr/bin/env python3

"""
Brings up gemma_audio_classification_node alone, configured to listen to an
AUDIO_VALIDATION_DATA_SET rosbag rather than to the live microphone.

Run this in one terminal, then drive it with
roboscout-assessment/audio_classification/scripts/run_all_bags.py in another.

Differences from spot_audio.launch.py, all of which are properties of the data
set and not of the model:
  - topic '/audio', not '/<SPOT_NAME>/raw_audio'
  - 48 kHz STEREO input (the bags carry 2 interleaved channels)
  - require_assessing false: these bags contain only AudioDataStamped, so the
    spot_status gate would suppress everything
  - no microphone or speaker node

Playback runs on the real clock, NOT `--clock`/use_sim_time. Windows advance on
accumulated sample count rather than on a timer (see audio_window_buffer.py), so
sim time buys nothing here -- while each bag starting at its own recorded epoch
would make sim time jump backwards at every bag boundary, which rclpy timers
handle badly.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DATASET_DIR = '/home/cdcl/cdcl_ws/drive2/datasets/AUDIO_VALIDATION_DATA_SET'


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument('model_id', default_value='google/gemma-4-12B-it',
                              description='Unified (audio-capable) Gemma 4 checkpoint. '
                                          'google/gemma-4-e4b-it is the smaller option; '
                                          'gemma-4-26b-a4b-it has no audio tower and will '
                                          'be rejected at load.'),
        DeclareLaunchArgument('audio_topic', default_value='/audio'),
        DeclareLaunchArgument('input_sample_rate', default_value='48000'),
        DeclareLaunchArgument('input_channels', default_value='2'),
        DeclareLaunchArgument('channel_mode', default_value='mix',
                              description="'mix' to average channels, or a channel index"),
        DeclareLaunchArgument('window_s', default_value='30.0',
                             description='Analysis window; 30 s is the model'
                                          "'s per-segment audio limit."),
        DeclareLaunchArgument('hop_s', default_value='15.0'),
        DeclareLaunchArgument('min_tail_s', default_value='1.0',
                              description='Shortest trailing remainder still classified. '
                                          'The data set has 16 clips under 5 s (shortest '
                                          '2.0 s) that only ever produce a tail window, so '
                                          'a larger value drops them from the evaluation '
                                          'without saying so.'),
        DeclareLaunchArgument('take_notes', default_value='true',
                              description='Take per-window observation notes and reach '
                                          'one whole-recording verdict from them. Required '
                                          'for the verbal-alertness rubric, which counts '
                                          'prompts and responses across the encounter. '
                                          'Adds ~2.5 s per window.'),
        DeclareLaunchArgument('ground_in_note', default_value='true',
                              description='Score each question conditioned on the '
                                          'window note rather than the audio alone. '
                                          'Free when take_notes is on.'),
        DeclareLaunchArgument('output_dir', default_value=f'{DATASET_DIR}/results'),
        DeclareLaunchArgument('labels_json', default_value=f'{DATASET_DIR}/labels.json'),
    ]

    node = Node(
        package='spot_audio',
        executable='gemma_audio_classification_node.py',
        name='gemma_audio_classification_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'model_id': LaunchConfiguration('model_id'),
            'audio_topic': LaunchConfiguration('audio_topic'),
            'input_sample_rate': LaunchConfiguration('input_sample_rate'),
            'input_channels': LaunchConfiguration('input_channels'),
            'channel_mode': LaunchConfiguration('channel_mode'),
            'window_s': LaunchConfiguration('window_s'),
            'hop_s': LaunchConfiguration('hop_s'),
            'min_tail_s': LaunchConfiguration('min_tail_s'),
            'take_notes': LaunchConfiguration('take_notes'),
            'ground_in_note': LaunchConfiguration('ground_in_note'),
            'require_assessing': False,
            'save_artifacts': True,
            'output_dir': LaunchConfiguration('output_dir'),
            'labels_json': LaunchConfiguration('labels_json'),
            'platform_name': 'eval',
        }],
    )

    return LaunchDescription(arguments + [node])
