#!/usr/bin/env python3


"""
This node takes raw audio being made available by the native PulseAudio microphone and publishes it for downstream tasks, such as transcription and classification.
"""

from audio_common_msgs.msg import AudioData, AudioDataStamped
from microphone.microphone_device import MicrophoneDevice, PulseMicrophoneDevice
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Header
import time
import os
import wave
import threading

class MicrophoneNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        # Main microphone parameters (The filtered AEC mic)
        self.declare_parameter('microphone_name', 'robot_aec_mic')
        self.declare_parameter('microphone_sampling_freq_hz', 48000) 
        self.declare_parameter('main_channel', 0)
        
        # Secondary debugging parameters
        self.declare_parameter('raw_microphone_name', 'alsa_input.usb-R__DE_Microphones_R__DE_VideoMic_NTG_C0ECAE69-00.analog-stereo') # The raw hardware mic string
        self.declare_parameter('save_data', False)
        self.declare_parameter('output_dir', '/tmp/mic_audio')

        self.pub_raw_audio = self.create_publisher(AudioDataStamped, 'raw_audio', 10)

        # Buffers for saving audio
        self.is_saving = self.get_parameter('save_data').value
        self.raw_mic_name = self.get_parameter('raw_microphone_name').value
        
        self.aec_audio_buffer = []
        self.raw_audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.wav_counter = 0

        # Create output directory if saving is enabled
        if self.is_saving:
            os.makedirs(self.get_parameter('output_dir').value, exist_ok=True)
            self.save_timer = self.create_timer(20.0, self.save_audio_timer_callback)

        # 1. Create main AEC microphone device
        self.microphone = PulseMicrophoneDevice(
            self.on_received_aec_audio,
            self.get_logger(),
            self.get_parameter('microphone_sampling_freq_hz').value
        )
        
        # 2. Create secondary Raw microphone device (only if saving and name is provided)
        self.raw_microphone = None
        if self.is_saving and self.raw_mic_name:
            self.raw_microphone = PulseMicrophoneDevice(
                self.on_received_raw_audio,
                self.get_logger(),
                self.get_parameter('microphone_sampling_freq_hz').value
            )

        self.seq = 0
        self.last_got_audio_data = None
        self.timer = self.create_timer(1.5, self.timer_callback) 

        self._connect()

    def _connect(self) -> None:
        # Connect main AEC stream
        if not self.microphone.find_device(self.get_parameter('microphone_name').value):
            self.get_logger().fatal("Failed to find main AEC PulseAudio source.")
            while True:
                time.sleep(5)

        # Connect secondary raw stream (if enabled)
        if self.raw_microphone is not None:
            if not self.raw_microphone.find_device(self.raw_mic_name):
                self.get_logger().error(f"Failed to find raw mic: {self.raw_mic_name}. Disabling raw recording.")
                self.raw_microphone = None

        try:
            self.microphone.start_stream()
            if self.raw_microphone is not None:
                self.raw_microphone.start_stream()
            self.get_logger().info("Successfully started audio streams!")
        except Exception as e:
            self.get_logger().fatal(f"Error starting streams: {e}")
            while True:
                time.sleep(5)

    def timer_callback(self) -> None:
        if self.last_got_audio_data is None:
            return
        
        if self.get_clock().now() - self.last_got_audio_data > Duration(seconds=1.0):
            self._disconnect()
            self._reconnect()

    def save_audio_timer_callback(self) -> None:
        with self.buffer_lock:
            if not self.aec_audio_buffer:
                return
            
            # Copy and clear buffers
            aec_to_save = np.array(self.aec_audio_buffer, dtype=np.int16)
            self.aec_audio_buffer.clear()
            
            raw_to_save = None
            if self.raw_microphone is not None and self.raw_audio_buffer:
                raw_to_save = np.array(self.raw_audio_buffer, dtype=np.int16)
                self.raw_audio_buffer.clear()

        output_dir = self.get_parameter('output_dir').value
        freq = self.get_parameter('microphone_sampling_freq_hz').value
        
        # Save AEC file
        aec_filename = os.path.join(output_dir, f"mic_aec_{self.wav_counter}.wav")
        self._write_wav(aec_filename, aec_to_save, freq)
        
        # Save Raw file
        if raw_to_save is not None:
            raw_filename = os.path.join(output_dir, f"mic_raw_{self.wav_counter}.wav")
            self._write_wav(raw_filename, raw_to_save, freq)
            
        self.get_logger().info(f"Saved synchronized debug audio chunk {self.wav_counter}.")
        self.wav_counter += 1

    def _write_wav(self, filename: str, data: np.ndarray, freq: int) -> None:
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(freq)
                wf.writeframes(data.tobytes())
        except Exception as e:
            self.get_logger().error(f"Failed to save debug audio {filename}: {e}")
    
    def _disconnect(self) -> None:
        try:
            self.microphone.stop_stream()
            if self.raw_microphone:
                self.raw_microphone.stop_stream()
        except Exception as e:
            self.get_logger().fatal(f"Error disconnecting from microphone: {e}.")

    def _reconnect(self) -> None:
        try:
            if self.microphone.find_device(self.get_parameter('microphone_name').value):
                self.microphone.start_stream()
            if self.raw_microphone and self.raw_microphone.find_device(self.raw_mic_name):
                self.raw_microphone.start_stream()
        except Exception as e:
            self.get_logger().fatal(f"Error reconnecting to microphone: {e}")

    def on_received_aec_audio(self, data, channel) -> None:
        if channel == self.get_parameter('main_channel').value:
            arr_int16 = np.array(data, dtype=np.int16)
            
            if self.is_saving:
                with self.buffer_lock:
                    self.aec_audio_buffer.extend(arr_int16.tolist())

            # Only the AEC audio gets published to ROS!
            self.pub_raw_audio.publish(
                AudioDataStamped(
                    header=Header(stamp=self.get_clock().now().to_msg()),
                    audio=AudioData(data=arr_int16.view(np.uint8).tolist())
                )
            )
            self.last_got_audio_data = self.get_clock().now()
            self.seq += 1

    def on_received_raw_audio(self, data, channel) -> None:
        if channel == self.get_parameter('main_channel').value and self.is_saving:
            arr_int16 = np.array(data, dtype=np.int16)
            with self.buffer_lock:
                self.raw_audio_buffer.extend(arr_int16.tolist())


def main(args=None) -> None:
    rclpy.init(args=args)
    microphone_node = MicrophoneNode('microphone_node')
    rclpy.spin(microphone_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()