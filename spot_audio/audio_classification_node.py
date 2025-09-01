#!/usr/bin/env python3


from audio_common_msgs.msg import AudioData
from cdcl_umd_msgs.msg import Observation, ObservationModule, ObservationDataSource, SpotStatus
from cdcl_umd_msgs.srv import StopListening
from faster_whisper import WhisperModel
import math
from microphone.audio_classifier import MaxArgStrategy
import noisereduce as nr
import numpy as np
import os
import random
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.signal import resample_poly
from std_msgs.msg import Empty
import time
import torch
from typing import Optional
import wave

spot_name = os.environ['SPOT_NAME']



class AudioClassificationNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        # the microphone parameters and microphone raw data
        self.declare_parameter('microphone_rate', 48000)
        self.sub_audio_data = self.create_subscription(
            AudioData,
            '/' + spot_name + '/raw_audio',
            self.audio_data_callback,
            10
        )

        # noise buffer contains noise sample from environment
        self.declare_parameter('noise_buffer_length_s', 5.0)  # length of noise buffer in seconds
        self.noise_buffer = None

        # rolling buffer contains the audio we transcribe
        self.declare_parameter('rolling_window_period_s', 5.0)
        self.rolling_samples = int(self.get_parameter('rolling_window_period_s').value * 16000)  # 16000 is the target sample rate
        self.rolling_buffer = np.zeros(self.rolling_samples, dtype=np.int16)

        # AST (audio classifier)
        self.declare_parameter('path_to_classifier', "/home/cdcl/cdcl_ws/models/alertness_verbal_classifiers/")
        self.audio_classification_strategy = MaxArgStrategy(
            self.get_parameter('path_to_classifier').value,
            self.get_logger()
        )
        self.audio_classification_strategy.load_classifier()

        # whisper (stt model)
        self.declare_parameter('path_to_whisper', '/home/cdcl/cdcl_ws/models/whisper/')
        self.transcriber = WhisperModel(
            self.get_parameter('path_to_whisper').value,
            device='cuda',
            compute_type='int8',
            local_files_only=False
        )

        # the speech we heard the person say + output of classifiers
        self.pub_speech = self.create_publisher(ObservationDataSource, 'speech', 10)
        self.finalized_speech_buffer = ""
        self.pub_observation = self.create_publisher(Observation, 'observation_no_id', 10)
        self.pub_observation_data_source = self.create_publisher(ObservationDataSource, 'observation_data_sources', 10)

        # we stop listening to the audio, when Spot is speaking
        self.stop_listening_service = self.create_service(StopListening, 'stop_listening', self.stop_listening_callback)
        self.stop_listening_start_time = None
        self.stop_listening_stop_time = None

        # whether we should save audio we record and where to save it
        self.seq = 0
        self.declare_parameter('path_to_saved_audio', '/home/cdcl/cdcl_ws/audio')
        self.declare_parameter('save_audio', False)

        # how frequently to process the audio buffer
        self.transcription_timer_period_s = 5.0
        self.transcription_timer = self.create_timer(
            self.transcription_timer_period_s,
            self.transcription_timer_callback
        )
        self.transcription_to_classification_period = 2  # classify 5x as frequently as we transcribe
        self.classification_timer = self.create_timer(
            self.transcription_timer_period_s / self.transcription_to_classification_period,
            self.classification_timer_callback
        )

        # only process audio while we are assessing
        self.sub_spot_status = self.create_subscription(SpotStatus, 'spot_status', self.spot_status_callback, qos_profile_sensor_data)
        self.assessing = False

        # heart beat stuff
        self.pub_heartbeat = self.create_publisher(
            Empty,
            'audio_classification/heartbeat',
            10
        )
        self.heartbeat_timer = self.create_timer(2.5, self.heartbeat_callback)

    def heartbeat_callback(self) -> None:
        """
        sends a heartbeat message so the spot status publisher knows if whisper + AST are working
        :return: nothing
        """
        self.pub_heartbeat.publish(Empty())

    @staticmethod
    def amplify_audio(buffered_audio: np.ndarray, gain: float=1.0) -> np.ndarray:
        """
        amplifies audio buffer by gain (w/out saturating)
        :param buffered_audio: audio in PCM 16 format mono
        :param gain: constant to multiply audio by
        :return: amplified audio
        """
        peak = np.max(np.abs(buffered_audio))
        gain = gain * float(2**15 - 1) / peak
        return np.clip(buffered_audio.astype(np.float32) * gain, -2**15, 2**15 - 1).astype(np.int16)

    @staticmethod
    def resample_int16(buffered_audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample a 1D int16 signal using polyphase filtering (high quality / fast).
        :param buffered_audio: audio in PCM 16 format mono (np.int16)
        :param orig_sr: original sample rate
        :param target_sr: target sample rate
        :return: resampled audio buffer in PCM 16 format mono (np.int16)
        """
        if orig_sr == target_sr:
            return buffered_audio
        g = math.gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        y = resample_poly(buffered_audio.astype(np.float32), up, down)
        return np.clip(y, -32768, 32767).astype(np.int16)

    @staticmethod
    def filter_audio(buffered_audio: np.ndarray, rate: int, noise_sample: Optional[np.ndarray]=None):
        """
        filters audio buffer using noisereduce package
        :param buffered_audio: the audio to be filtered (PCM 16, np.int16, mono)
        :param rate: the rate at which audio is sampled
        :param noise_sample: a sample of the noise (PCM 16, np.int16, mono)
        :return: filtered audio  (PCM 16, np.int16, mono)
        """
        return nr.reduce_noise(
            y=buffered_audio,
            y_noise=noise_sample,
            sr=rate,
            stationary=True,
            prop_decrease=0.8,
            n_fft=512,
        ).astype(np.int16)

    @staticmethod
    def audio_length(buffered_audio: np.ndarray, rate: float) -> float:
        """
        computes length in seconds of audio snippet given its rate
        :param buffered_audio: audio sample (PCM 16, np.int16, mono)
        :param rate: the rate at which the audio is sampled
        :return: nothing
        """
        return len(buffered_audio) / rate

    def save_audio(self, buffered_audio: np.ndarray, path: str, file_name: str, rate: int) -> None:
        """
        saves an audio segment to a provided path / filename as a .wav file
        :param buffered_audio: the audio (PCM 16, mono)
        :param path: path to save audio .wav file
        :param file_name: name of file
        :param rate: rate at which audio is sampled
        :return: nothing
        """
        wav_path = os.path.join(path, file_name)
        self.seq += 1
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(buffered_audio.tobytes())

    def spot_status_callback(self, msg: SpotStatus) -> None:
        """
        sets self.assessing to true if we are assessing, otherwise it's set to false
        :param msg:  the spot status message
        :return: nothing
        """
        self.assessing = msg.state == SpotStatus.ASSESSING

    def stop_listening_callback(self, request, response):
        """
        assigns values "self.stop_listening_start_time" and "self.stop_listening_stop_time", which the node will use
        to ignore any audio heard within those two time stamps
        :param request: the ros2 request
        :param response: the response
        :return:
        """
        self.get_logger().info(f"Processing request to stop listening.")
        self.stop_listening_start_time = Time.from_msg(request.stop_listen_time)
        self.stop_listening_stop_time = Time.from_msg(request.start_listen_time)
        response.success = True
        self.get_logger().info(f"Processed request to stop listening. Stop Listening Starting: {self.stop_listening_start_time}, Stop Listening Ending: {self.stop_listening_stop_time}")
        return response

    def classification_timer_callback(self) -> None:
        """
        classifies the audio buffer
        :return: nothing
        """
        # only process audio when we are assessing
        if not self.assessing:
            return

        noise_buffer_length_s = AudioClassificationNode.audio_length(self.noise_buffer, 16000.0)
        rolling_buffer_length_s = AudioClassificationNode.audio_length(self.rolling_buffer, 16000.0)
        noise_buffer_full = noise_buffer_length_s >= self.get_parameter('noise_buffer_length_s').value
        rolling_buffer_full = rolling_buffer_length_s >= self.get_parameter('rolling_window_period_s').value
        if noise_buffer_full and rolling_buffer_full:
            self.get_logger().debug(f'Before processing, buffer length (s): {rolling_buffer_length_s}')

            # classify the audio (classify 1 second chunks)
            start = time.time()
            chunks = np.array_split(self.rolling_buffer, self.transcription_to_classification_period)
            classification_tensor = self.audio_classification_strategy.classify_audio(
                chunks[-1]
                # AudioClassificationNode.pcm_to_f32(chunks[-1])
            )
            classifications = self.audio_classification_strategy.apply_strategy(classification_tensor)

            # if we got a valid output, then publish observation + observation data source messages
            if classifications is not None:
                respiratory_distress_classification, verbal_alertness_classification = classifications
                respiratory_distress_label = torch.argmax(respiratory_distress_classification).item()
                verbal_alertness_label = torch.argmax(verbal_alertness_classification).item()

                data_source = ObservationDataSource(
                    data_source_id=random.randint(-2**31, 2**31 - 1),
                    raw_audio=self.rolling_buffer.view(np.uint8).tolist(),
                    platform_name=spot_name
                )
                published_observation = False
                if (verbal_alertness_label == 1 or verbal_alertness_label == 0) and not torch.isnan(verbal_alertness_classification).any().item():
                    alertness_verbal_observation = Observation(
                        stamp=self.get_clock().now().to_msg(),
                        platform_name=spot_name,
                        data_source_id=data_source.data_source_id,
                        observation_module = ObservationModule.AST_ALERTNESS_VERBAL,
                        observation=verbal_alertness_classification.tolist()
                    )
                    self.pub_observation.publish(alertness_verbal_observation)
                    published_observation = True
                if respiratory_distress_label == 1 and  not torch.isnan(respiratory_distress_classification).any().item():
                    respiratory_distress_observation = Observation(
                        stamp=self.get_clock().now().to_msg(),
                        platform_name=spot_name,
                        data_source_id=data_source.data_source_id,
                        observation_module = ObservationModule.AST_RESPIRATORY_DISTRESS,
                        observation=respiratory_distress_classification.tolist()
                    )
                    self.pub_observation.publish(respiratory_distress_observation)
                    published_observation = True

                if published_observation:
                    self.pub_observation_data_source.publish(data_source)

                # print out the label for the user
                self.get_logger().debug(f"Respiratory Distress: {respiratory_distress_label}, Verbal Alertness: {verbal_alertness_label}")
            stop = time.time()
            self.get_logger().debug(f"Took {stop-start:.2f} s to classify audio.")

    def transcription_timer_callback(self) -> None:
        """
        transcribes the audio buffer
        :return: nothing
        """
        # only process audio when we are assessing
        if not self.assessing:
            return

        # if noise and rolling buffers are both full, then we can proceed with processing the rolling window buffer
        noise_buffer_length_s = AudioClassificationNode.audio_length(self.noise_buffer, 16000.0)
        rolling_buffer_length_s = AudioClassificationNode.audio_length(self.rolling_buffer, 16000.0)
        noise_buffer_full = noise_buffer_length_s >= self.get_parameter('noise_buffer_length_s').value
        rolling_buffer_full = rolling_buffer_length_s >= self.get_parameter('rolling_window_period_s').value
        if noise_buffer_full and rolling_buffer_full:
            self.get_logger().debug(f'Before processing, buffer length (s): {rolling_buffer_length_s}')

            # save the audio file for debugging before transcription
            if self.get_parameter('save_audio').value:
                self.save_audio(
                    self.rolling_buffer,
                    self.get_parameter('path_to_saved_audio').value,
                    f'{self.seq}.wav',
                    16000
                )

            # transcribe the rolling buffer, timing how long it takes
            start = time.time()
            segments_gen, info = self.transcriber.transcribe(
                AudioClassificationNode.pcm_to_f32(self.rolling_buffer),
                language='en',
                task='transcribe',
                temperature=0.1  # reducing => fewer hallucinations
            )
            stop = time.time()
            self.get_logger().debug(f'Took {stop - start:.2f} s to transcribe audio.')  # takes 0.05-0.07 seconds on HP

            # add speech we are confident in to "finalized speech buffer" for later publishing
            for segment in segments_gen:
                prob_speech = 1.0 - segment.no_speech_prob
                self.get_logger().debug(f"     [{segment.start:.2f} - {segment.end:.2f}] [{prob_speech:.3f}] {segment.text}")
                if prob_speech > 0.7:
                    self.finalized_speech_buffer += segment.text

            # Publish the audio we are confident in, then clear the speech buffer
            self.get_logger().debug(f'Speech: {self.finalized_speech_buffer}')
            if len(self.finalized_speech_buffer) != 0:
                speech = ObservationDataSource(
                    data_source_id=random.randint(-2**31, 2**31 - 1),
                    raw_audio=self.rolling_buffer.view(np.uint8).tolist(),
                    platform_name=spot_name,
                    audio_transcript=self.finalized_speech_buffer
                )
                self.pub_speech.publish(speech)
            self.finalized_speech_buffer = ""


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
        resamples audio to be 16000 Hz, amplifies and filters audio, then appends clean audio to either the noise buffer
        or a rolling buffer used for assessment
        :param msg: raw audio snippet
        :return: nothing
        """

        buffered_audio_u8 = np.array(msg.data, dtype=np.uint8)
        buffered_audio_16 = buffered_audio_u8.view(np.int16)
        buffered_audio_16 = AudioClassificationNode.resample_int16(buffered_audio_16, self.get_parameter('microphone_rate').value, 16000)
        amplified_buffered_audio_16 = AudioClassificationNode.amplify_audio(buffered_audio_16, 1.4)

        # add microphone audio to noise buffer till noise buffer full
        if self.noise_buffer is None:
            self.noise_buffer = amplified_buffered_audio_16
            return
        elif AudioClassificationNode.audio_length(self.noise_buffer, 16000.0) < self.get_parameter('noise_buffer_length_s').value:
            self.noise_buffer = np.concatenate([self.noise_buffer, amplified_buffered_audio_16])
            self.get_logger().debug(f"Noise Buffer (s): {AudioClassificationNode.audio_length(self.noise_buffer, 16000.0)}")
            return
        # noise buffer full here

        # save the noise buffer
        if self.get_parameter('save_audio').value:
            self.save_audio(self.noise_buffer, self.get_parameter('path_to_saved_audio').value,  f'noise.wav', 16000)


        # ignore any audio received while we are speaking, otherwise, filter and amplify the audio
        if self.stop_listening_start_time is not None and self.stop_listening_stop_time is not None and self.stop_listening_start_time <= self.get_clock().now() <= self.stop_listening_stop_time:
            filtered_amplified_buffered_audio_16 = np.zeros_like(amplified_buffered_audio_16)
        else:
            filtered_amplified_buffered_audio_16 = AudioClassificationNode.filter_audio(
                amplified_buffered_audio_16,
                16000,
                noise_sample=self.noise_buffer
            )

        # append the filtered and amplified audio to the buffer for later processing
        self.rolling_buffer = np.concatenate([self.rolling_buffer, filtered_amplified_buffered_audio_16])[-self.rolling_samples:]

def main(args=None) -> None:
    rclpy.init(args=args)
    audio_classification_node = AudioClassificationNode('audio_classification_node')
    rclpy.spin(audio_classification_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
