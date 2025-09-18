#!/usr/bin/env python3


"""

This node takes raw audio being made available by the microphone and publishes it for downstream tasks, such as
transcription and classification.

"""


from audio_common_msgs.msg import AudioData, AudioDataStamped
from microphone.microphone_device import MicrophoneDevice
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Header
import time


class MicrophoneNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        # declare microphone parameters
        self.declare_parameter('microphone_name', 'RØDE')
        self.declare_parameter('microphone_sampling_freq_hz', 48000)  # 48000 for rode and 16000 for respeaker
        self.declare_parameter('main_channel', 0)
        self.pub_raw_audio = self.create_publisher(AudioDataStamped, 'raw_audio', 10)

        # create a microphone device
        self.microphone = MicrophoneDevice(
            self.on_received_audio,
            self.get_logger(),
            self.get_parameter('microphone_sampling_freq_hz').value
        )
        self.microphone.create_pyaudio()
        self.seq = 0
        self.last_got_audio_data = None

        # watches to see if we haven't gotten data from device in a few seconds
        self.timer = self.create_timer(3.0, self.timer_callback)  # watch

        # wait in an infinite loop (rather than crashing the node) if we cannot connect to device
        if not self.microphone.find_device(self.get_parameter('microphone_name').value):
            self.get_logger().fatal("Failed to find audio device with matching name.")
            while True:
                time.sleep(5)

        # start streaming audio data immediately
        try:
            self.microphone.create_stream()
            self.microphone.start_stream()
            self.get_logger().info(f"Successfully streaming audio from {self.get_parameter('microphone_name').value} mic!")
        except Exception as e:
            self.get_logger().fatal(f"Encountered error while creating/starting stream. Error was: {e}")
            while True:
                time.sleep(5)

    def timer_callback(self) -> None:
        if self.last_got_audio_data is None:
            return
        
        # try restarting stream, if we haven't heard from mic in awhile
        if self.get_clock().now() - self.last_got_audio_data > Duration(seconds=5.0):
            self._disconnect()
            self._reconnect()
    
    def _disconnect(self) -> None:
        try:
            self.microphone.stop_stream()
            self.microphone.destroy_stream()
            self.microphone.destroy_pyaudio()
        except Exception as e:
            self.get_logger().fatal(f"Encountered error while disconnecting from microphone. Error was {e}.")

    def _reconnect(self) -> None:
        try:
            self.microphone.create_pyaudio()
            if not self.microphone.find_device(self.get_parameter('microphone_name').value):
                return
            self.microphone.create_stream()
            self.microphone.start_stream()
        except Exception as e:
            self.get_logger().fatal(f"Encountered error while reconnecting to microphone. Error was {e}")
        
        



    def on_received_audio(self, data, channel) -> None:
        """
        publishes raw audio for downstream tasks
        :param data: new microphone data
        :param channel: the channel providing the data
        :return: nothing
        """
        if channel == self.get_parameter('main_channel').value:
            # RØDE microphone publishes PCM 16 format audio data and AudioData requires raw uint8 bytes, publish
            arr_int16 = np.array(data, dtype=np.int16)
            self.pub_raw_audio.publish(
                AudioDataStamped(
                    header=Header(
                        stamp=self.get_clock().now().to_msg()
                    ),
                    audio=AudioData(
                        data=arr_int16.view(np.uint8).tolist()
                    )
                )
            )
            self.last_got_audio_data = self.get_clock().now()
            self.seq += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    microphone_node = MicrophoneNode('microphone_node')
    rclpy.spin(microphone_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
