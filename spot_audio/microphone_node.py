#!/usr/bin/env python3


"""

This node takes raw audio being made available by the microphone and publishes it for downstream tasks, such as
transcription and classification.

"""


from audio_common_msgs.msg import AudioData, AudioDataStamped
from microphone.udp_microphone_device import UdpMicrophoneDevice
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Header
import time
import wave



class MicrophoneNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        # declare microphone parameters
        self.declare_parameter('microphone_udp_port', 21885)
        self.declare_parameter('microphone_sampling_freq_hz', 48000)  # 48000 for rode and 16000 for respeaker
        self.declare_parameter('main_channel', 0)
        self.pub_raw_audio = self.create_publisher(AudioDataStamped, 'raw_audio', 10)

        # for saving audio to .wav
        self.save_debug_audio = True
        self.audio_buffer = bytearray()
        self.record_start_time = None


        # create a microphone device
        self.microphone = UdpMicrophoneDevice(
            self.on_received_audio,
            self.get_logger(),
            self.get_parameter('microphone_sampling_freq_hz').value,
            self.get_parameter('microphone_udp_port').value
        )

        self.seq = 0
        self.last_got_audio_data = None

        # watches to see if we haven't gotten data from device in a few seconds
        self.timer = self.create_timer(1.5, self.timer_callback)  # watch

        # start streaming audio data immediately
        try:
            self.microphone.start_stream()
            self.get_logger().info(f"Started streaming audio from RODE mic!")
        except Exception as e:
            self.get_logger().fatal(f"Encountered error while creating/starting stream. Error was: {e}")
            while True:
                time.sleep(5)

    def timer_callback(self) -> None:
        if self.last_got_audio_data is None:
            return
        
        # try restarting stream, if we haven't heard from mic in awhile
        if self.get_clock().now() - self.last_got_audio_data > Duration(seconds=1.0):
            self._disconnect()
            self._reconnect()
    
    def _save_to_wav(self) -> None:
        filename = "/home/cdcl/cdcl_ws/debug_audio_10s.wav"
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # Assuming mono based on main_channel logic
                wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
                wf.setframerate(self.get_parameter('microphone_sampling_freq_hz').value)
                wf.writeframes(self.audio_buffer)
            self.get_logger().info(f"Successfully saved 10 seconds of raw audio to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save debug audio file: {e}")
        finally:
            # Free up memory once saved
            self.audio_buffer.clear()

    def _disconnect(self) -> None:
        try:
            self.microphone.stop_stream()
        except Exception as e:
            self.get_logger().fatal(f"Encountered error while disconnecting from microphone. Error was {e}.")

    def _reconnect(self) -> None:
        try:
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

            # NEW: 10-second debug recording logic
            if self.save_debug_audio:
                if self.record_start_time is None:
                    self.record_start_time = self.get_clock().now()
                
                # Append raw bytes to our buffer
                self.audio_buffer.extend(arr_int16.tobytes())

                # Check if 10 seconds have elapsed
                if self.get_clock().now() - self.record_start_time > Duration(seconds=10.0):
                    self._save_to_wav()
                    self.save_debug_audio = False  # Ensure we only do this once

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
