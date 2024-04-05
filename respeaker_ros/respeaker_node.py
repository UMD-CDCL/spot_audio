#!/usr/bin/env python3
import usb.core
import usb.util

import rclpy
from rclpy.node import Node
import rclpy.time
import rclpy.duration
import rclpy.timer
from rclpy.qos import QoSProfile, QoSDurabilityPolicy


from audio_common_msgs.msg import AudioData
# from std_msgs.msg import UInt16, Int32, Bool, ColorRGBA
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Pose
from cdcl_umd_msgs.msg import Speech

from respeaker_ros.interface import RespeakerInterface
from respeaker_ros.audio import RespeakerAudio
import math
import numpy as np
import angles
import tf_transformations as T

from respeaker_ros.transcriber import WhisperModel


class RespeakerNode(Node):
    def __init__(self):
        super().__init__('respeaker_node')

        self.sensor_frame_id = self.declare_parameter('sensor_frame_id', 'respeaker_base')
        self.speech_prefetch = self.declare_parameter('speech_prefetch', 0.5)
        self.update_period_s = self.declare_parameter('update_period_s', 0.1)
        self.main_channel = self.declare_parameter('main_channel', 0)
        self.speech_continuation = self.declare_parameter('speech_continuation', 0.75)
        self.speech_max_duration = self.declare_parameter('speech_max_duration', 8.0)
        self.speech_min_duration = self.declare_parameter('speech_min_duration', 0.1)
        self.doa_yaw_offset = self.declare_parameter('doa_yaw_offset', 90.0)

        self.respeaker = RespeakerInterface()
        self.respeaker_audio = RespeakerAudio(self.on_audio, suppress_error=True)

        self.speech_audio_buffer = []
        self.is_speaking = False
        self.speech_stopped = self.get_clock().now()
        self.prev_is_voice = None
        self.prev_doa = None
        latching_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self._pub_audio = self.create_publisher(AudioData, 'audio', 10)
        self._pub_speech = self.create_publisher(Speech, 'speech_audio', 10)
        self._timer = self.create_timer(self.update_period_s.value, self.on_timer)

        self.speech_prefetch_bytes = int(
            self.speech_prefetch.value * self.respeaker_audio.rate * self.respeaker_audio.bitdepth / 8.0)
        self.speech_prefetch_buffer = np.zeros(self.speech_prefetch_bytes, dtype=np.uint8)
        self.respeaker_audio.start()
 
        self.timer_led = None
        self.sub_led = self.create_subscription(ColorRGBA, "status_led", self.on_status_led, 1)


        # speech-to-text stuff
        self.model_sizes = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v2", "large-v3",
        ]
        self.model_size = 'base.en'
        self.language = 'en'
        self.task = 'transcribe'
        self.initial_prompt = None
        self.vad_parameters={'threshold': 0.5}
        self.use_vad=True
        self.device = 'cpu'

        self.transcriber = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type='int8',
            local_files_only=False
        )


    def on_audio(self, data, channel):
        if channel == self.main_channel.value:
            if self._pub_audio.get_subscription_count() > 0:
                self._pub_audio.publish(AudioData(data=data.astype(np.uint8)))
            if self.is_speaking:
                if len(self.speech_audio_buffer) == 0:
                    self.speech_audio_buffer = [self.speech_prefetch_buffer]
                self.speech_audio_buffer.append(data)
            else:
                self.speech_prefetch_buffer = np.roll(self.speech_prefetch_buffer, -len(data))
                self.speech_prefetch_buffer[-len(data):] = data

    def on_timer(self):
        stamp = self.get_clock().now()
        is_voice = self.respeaker.is_voice()
        doa_rad = math.radians(self.respeaker.direction - 180.0)
        doa_rad = angles.shortest_angular_distance(
            doa_rad, math.radians(self.doa_yaw_offset.value))
        doa = math.degrees(doa_rad)

        # vad
        if is_voice != self.prev_is_voice:
            self.prev_is_voice = is_voice

        # doa
        if doa != self.prev_doa:
            self.prev_doa = doa

        # speech audio
        if is_voice:
            self.speech_stopped = stamp
        if stamp - self.speech_stopped < rclpy.duration.Duration(nanoseconds=self.speech_continuation.value * 1e9):
            self.is_speaking = True
        elif self.is_speaking:
            buffered_speech = self.speech_audio_buffer
            self.speech_audio_buffer = []
            self.is_speaking = False
            if len(buffered_speech) == 0:
                return
            buffered_speech = np.hstack(buffered_speech)
            duration = 16 * len(buffered_speech) * self.respeaker_audio.bitwidth
            duration = duration / self.respeaker_audio.rate / self.respeaker_audio.bitdepth
            print("Speech detected for %.3f seconds" % duration)
            if self.speech_min_duration.value <= duration < self.speech_max_duration.value:
                if self._pub_speech.get_subscription_count() > 0:
                    speech_msg = Speech()
                    speech_msg.header.frame_id = self.sensor_frame_id.value
                    speech_msg.header.stamp = stamp.to_msg()

                    # use direction-of-arrival estimate to compute quaternion of speaker in microphone frame
                    ori = T.quaternion_from_euler(math.radians(doa), 0, 0)
                    speech_msg.doa.position.x = 0.0
                    speech_msg.doa.position.y = 0.0
                    speech_msg.doa.orientation.w = ori[0]
                    speech_msg.doa.orientation.x = ori[1]
                    speech_msg.doa.orientation.y = ori[2]
                    speech_msg.doa.orientation.z = ori[3]

                    # put raw speech data into message
                    speech_msg.raw_audio = buffered_speech.tolist()

                    # transcribe the speech and put in message
                    speech_msg.transcript = self.transcribe_speech(buffered_speech)

                    # finally publish the speech message
                    self._pub_speech.publish(speech_msg)

    def on_status_led(self, msg):
        self.respeaker.set_led_color(r=msg.r, g=msg.g, b=msg.b, a=msg.a)
        if self.timer_led and self.timer_led.is_alive():
            self.timer_led.destroy()
        self.timer_led = rclpy.timer.Timer(rclpy.duration.Duration(3.0),
                                     lambda e: self.respeaker.set_led_trace(),
                                     oneshot=True)
    
    def transcribe_speech(self, buffered_speech_int16):
        self.get_logger().info("Transcribing speech now...")

        # note you cannot send np.uint8, you must send a float32 as per whisper
        buffered_speech_float32 = self.bytes_to_float_array(buffered_speech_int16)
        result, info = self.transcriber.transcribe(
            buffered_speech_float32,
            initial_prompt=self.initial_prompt,
            language=self.language,
            task=self.task,
            vad_filter=False,
            vad_parameters=None
            # vad_filter=self.use_vad,
            # vad_parameters=self.vad_parameters if self.use_vad else None
        )

        if result:
            return result[0].text
        return ""
        # self.get_logger().warn('Result: {}'.format(result))
        # self.get_logger().warn('Info: {}'.format(info))

    @staticmethod
    def bytes_to_float_array(audio_int16):
        """
        Convert audio data from bytes to a NumPy float array.

        It assumes that the audio data is in 16-bit PCM format. The audio data is normalized to
        have values between -1 and 1.

        Args:
            audio_uint8 (bytes): Audio data in bytes.

        Returns:
            np.ndarray: A NumPy array containing the audio data as float values normalized between -1 and 1.
        """
        raw_data = np.frombuffer(buffer=audio_int16, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0


def main(args=None):
    rclpy.init(args=args)

    publisher = RespeakerNode()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass

    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
