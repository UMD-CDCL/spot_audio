#!/usr/bin/env python3


from audio_common_msgs.msg import AudioData
from cdcl_umd_msgs.msg import Observation, ObservationModule, ObservationDataSource, SpotStatus
import math
import numpy as np
import os
import rclpy
from rclpy.node import Node
from scipy.signal import butter, lfilter,  resample_poly, wiener


spot_name = os.environ['SPOT_NAME']



class AudioClassificationNode(Node):
    def __init__(self, node_name: str):
        """
        According to nav2 documentation (https://docs.nav2.org/concepts/index.html#lifecycle-nodes-and-bond), the node's
        constructor should not contain any ROS networking setup or parameter reading; it should simply declare all the
         member variables
        :param node_name: the name of the node
        """
        # call the super node
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.declare_parameter('microphone_rate', 48000)
        self.sub_audio_data = self.create_subscription(
            AudioData,
            '/' + spot_name + '/raw_audio',
            self.audio_data_callback,
            10
        )

        self.declare_parameter('noise_buffer_length_s', 5.0)
        self.noise_buffer = None

        self.declare_parameter('rolling_window_period_s', 30.0)
        self.rolling_samples = self.get_parameter('rolling_window_period_s').value * self.get_parameter('microphone_rate').value
        self.rolling_buffer = np.zeros(self.rolling_samples, dtype=np.int16)
        self.last_transcript_end_time: float = 0.0  #
        self.print_counter = 0



    @staticmethod
    def amplify_audio(buffered_audio, gain=1):
        peak = np.max(np.abs(buffered_audio))
        gain = gain * float(2**15 - 1) / peak
        return np.clip(buffered_audio.astype(np.float32) * gain, -2**15, 2**15 - 1).astype(np.int16)

    @staticmethod
    def resample_int16(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample a 1D int16 signal using polyphase filtering (high quality / fast).

        Args:
            x: np.int16 mono audio.
            orig_sr: original sample rate.
            target_sr: target sample rate.

        Returns:
            np.int16 resampled audio.
        """
        if orig_sr == target_sr:
            return x
        g = math.gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        y = resample_poly(x.astype(np.float32), up, down)
        return np.clip(y, -32768, 32767).astype(np.int16)

    @staticmethod
    def audio_length(buffer: np.ndarray, bitdepth: int, rate: float):
        return len(buffer) / (rate * bitdepth / 8.0)

    def audio_data_callback(self, msg: AudioData) -> None:
        """

        :param msg:
        :return:
        """

        buffered_audio_16 = AudioClassificationNode.resample_int16(np.array(msg.data), self.get_parameter('microphone_rate').value, 16000)
        buffer_length_s = AudioClassificationNode.audio_length(buffered_audio_16, bitdepth=16, rate=16000.0)

        # if noise buffer is shorter than the noise buffer length, add to noise buffer
        if self.noise_buffer is None:
            self.noise_buffer = buffered_audio_16
            return
        elif AudioClassificationNode.audio_length(self.noise_buffer, 16, 16000.0) < self.get_parameter('noise_buffer_length_s').value:
            self.noise_buffer = np.concatenate([self.noise_buffer, buffered_audio_16])
            self.get_logger().info(f"Noise Buffer (s): {AudioClassificationNode.audio_length(self.noise_buffer, 16, 16000.0)}")
            return
        self.get_logger().info(f'We got a noise buffer!')





    # def _transcribe_speech(self, audio_data: npt.NDArray) -> dict[str, int | None] | None:
    #     audio_data_f32 = pcm_to_f32(audio_data)
    #     result, info = self.transcriber.transcribe(
    #         audio_data_f32,
    #         language='en',
    #         task='transcribe'
    #     )
    #
    #     return result, info


def main(args=None) -> None:
    rclpy.init(args=args)
    audio_classification_node = AudioClassificationNode('audio_classification_node')
    rclpy.spin(audio_classification_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
