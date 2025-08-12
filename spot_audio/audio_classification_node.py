#!/usr/bin/env python3


from audio_common_msgs.msg import AudioData
from cdcl_umd_msgs.msg import Observation, ObservationModule, ObservationDataSource, SpotStatus
from faster_whisper import WhisperModel
import math
from microphone.audio_classifier import MaxArgStrategy
import noisereduce as nr
import numpy as np
import os
import rclpy
from rclpy.node import Node
from scipy.signal import butter, lfilter,  resample_poly, wiener
import time
import wave

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

        #
        self.declare_parameter('noise_buffer_length_s', 5.0)
        self.noise_buffer = None

        #
        self.declare_parameter('rolling_window_period_s', 10.0)
        self.rolling_samples = int(self.get_parameter('rolling_window_period_s').value * 16000)  # 16000 is the target sample rate
        self.rolling_buffer = np.zeros(self.rolling_samples, dtype=np.int16)
        self.last_transcript_end_time: float = 0.0  #  

        # try loading classifier model, if we fail, then return FAILURE, otherwise continue
        self.declare_parameter('path_to_classifier', "/home/cdcl/cdcl_ws/models/alertness_verbal_classifiers/")
        self.audio_classification_strategy = MaxArgStrategy(self.get_parameter('path_to_classifier').value, self.get_logger())
        self.audio_classification_strategy.load_classifier()
        self.device = 'cuda'
        self.declare_parameter('path_to_whisper', '/home/cdcl/cdcl_ws/models/whisper/')
        self.transcriber = WhisperModel(self.get_parameter('path_to_whisper').value, device=self.device, compute_type='int8', local_files_only=False)
        self.seq = 0
        self.declare_parameter('path_to_saved_audio', '/home/cdcl/cdcl_ws/audio')
        self.declare_parameter('save_audio', False)

        self.timer = self.create_timer(
            3.0,
            self.timer_callback
        )


    @staticmethod
    def amplify_audio(buffered_audio, gain: float=1.0):
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
    def filter_audio(buffered_audio, rate, noise_sample=None):
        return nr.reduce_noise(
            y=buffered_audio,
            y_noise=noise_sample,
            sr=rate,
            stationary=True,
            prop_decrease=0.8,
            n_fft=512,
            #time_mask_smooth_ms = 65
            #freq_mask_smooth_hz = 400
        ).astype(np.int16)

    @staticmethod
    def audio_length(buffer: np.ndarray, bitdepth: int, rate: float):
        # return len(buffer) / (rate * bitdepth / 8.0)
        return len(buffer) / (rate)

    def save_audio(self, buffer, path: str, file_name: str, rate):
        wav_path = os.path.join(path, file_name)
        self.seq += 1
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(buffer.tobytes())

    def timer_callback(self) -> None:

        noise_buffer_full = AudioClassificationNode.audio_length(self.noise_buffer, 16, 16000.0) >= self.get_parameter('noise_buffer_length_s').value

        rolling_buffer_length_s = AudioClassificationNode.audio_length(self.rolling_buffer, 16, 16000.0)
        rolling_buffer_full = rolling_buffer_length_s >= self.get_parameter('rolling_window_period_s').value
        if noise_buffer_full and rolling_buffer_full:
            self.get_logger().info(f'Before transcribing, buffer length (s): {AudioClassificationNode.audio_length(self.rolling_buffer, 16, 16000.0)}')

            # save the audio file for debugging before transcription
            if self.get_parameter('save_audio').value:
                self.save_audio(self.rolling_buffer, self.get_parameter('path_to_saved_audio').value, f'{self.seq}.wav', 16000)

            self.get_logger().info(f'Attempting to transcribe...')
            start = time.time()
            segments_gen, info = self.transcriber.transcribe(
                AudioClassificationNode.pcm_to_f32(self.rolling_buffer),
                language='en',
                task='transcribe',
                temperature=0.1
            )
            stop = time.time()
            self.get_logger().info(f'Took {stop - start:.2f} s to transcribe.')

            # print the transcription

            for segment in segments_gen:
                prob_speech = 1.0 - segment.no_speech_prob
                self.get_logger().info(f"[{segment.start:.2f} - {segment.end:.2f}] [{prob_speech:.3f}] {segment.text}")

                # if prob_speech > 0.5:
                #     self.get_logger().info(f"[{segment.start:.2f} - {segment.end:.2f}] [{1.0-segment.no_speech_prob:3f}] {segment.text}")


    @staticmethod
    def pcm_to_f32(raw_audio, bit_depth: type = np.int16):
        """
        convert PCM audio data from bytes to a numpy float array normalized to +/- 1.0
        :param raw_audio: the raw audio buffer
        :param bit_depth: the bit depth of the PCM signal (usually 16)
        :return: A numpy array containing the audio data as float values normalized between -1 and 1.
        """
        raw_data = np.frombuffer(buffer=raw_audio, dtype=bit_depth)
        max_amplitude = float(2 ** (bit_depth(0).nbytes * 8 - 1))  # 16 => 32768.0 (2^(16-1))
        return raw_data.astype(np.float32) / max_amplitude


    def audio_data_callback(self, msg: AudioData) -> None:
        """

        :param msg:
        :return:
        """

        buffered_audio_u8 = np.array(msg.data, dtype=np.uint8)
        buffered_audio_16 = buffered_audio_u8.view(np.int16)
        buffered_audio_16 = AudioClassificationNode.resample_int16(buffered_audio_16, self.get_parameter('microphone_rate').value, 16000)
        amplified_buffered_audio_16 = AudioClassificationNode.amplify_audio(buffered_audio_16, 1.4)

        # add microphone audio to noise buffer till noise buffer full
        if self.noise_buffer is None:
            self.noise_buffer = amplified_buffered_audio_16
            return
        elif AudioClassificationNode.audio_length(self.noise_buffer, 16, 16000.0) < self.get_parameter('noise_buffer_length_s').value:
            self.noise_buffer = np.concatenate([self.noise_buffer, amplified_buffered_audio_16])
            self.get_logger().debug(f"Noise Buffer (s): {AudioClassificationNode.audio_length(self.noise_buffer, 16, 16000.0)}")
            return
        # noise buffer full here
        self.save_audio(self.noise_buffer, self.get_parameter('path_to_saved_audio').value,  f'noise.wav', 16000)


        # add amplified, filtered microphone audio to rolling buffer
        filtered_amplified_buffered_audio_16 = AudioClassificationNode.filter_audio(
            amplified_buffered_audio_16,
            16000,
            noise_sample=self.noise_buffer
        )
        self.rolling_buffer = np.concatenate([self.rolling_buffer, filtered_amplified_buffered_audio_16])[-self.rolling_samples:]

        # # once rolling buffer is full, transcribe?
        # rolling_buffer_length_s = AudioClassificationNode.audio_length(self.rolling_buffer, 16, 16000.0)
        # rolling_buffer_full = rolling_buffer_length_s <= self.get_parameter('rolling_window_period_s').value
        # self.get_logger().info(f"Rolling buffer full? {rolling_buffer_full}, Rolling buffer length (s): {rolling_buffer_length_s}")
        # if rolling_buffer_full:
        #     self.get_logger().info(f'Before transcribing, buffer length (s): {AudioClassificationNode.audio_length(self.rolling_buffer, 16, 16000.0)}')
        #
        #     # save the audio file for debugging before transcription
        #     self.save_audio(self.rolling_buffer, self.get_parameter('path_to_saved_audio').value, f'{self.seq}.wav', 16000)
        #
        #     segments_gen, info = self.transcriber.transcribe(
        #         self.rolling_buffer,
        #         language='en',
        #         task='transcribe'
        #     )
        #
        #     for segment in segments_gen:
        #         self.get_logger().info(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}")
        #
        #     # self.rolling_buffer = self.rolling_buffer[int(16000 * 3):]
        #     # self.get_logger().info(f'After transcribing, buffer length (s): {AudioClassificationNode.audio_length(self.rolling_buffer, 16, 16000.0)}')







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
