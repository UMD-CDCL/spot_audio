#!/usr/bin/env python3


from audio_common_msgs.msg import AudioData
from cdcl_umd_msgs.msg import Observation, ObservationModule, ObservationDataSource, SpotStatus
from cdcl_umd_msgs.srv import StopListening
import diagnostic_updater
import diagnostic_msgs
from microphone.audio_classifier import MaxArgStrategy
from microphone.microphone_device import MicrophoneDevice
from microphone.transcriber import WhisperModel
from microphone.util import pcm_to_f32, save_audio_to_wav
import noisereduce as nr
import numpy as np
import numpy.typing as npt
import os
import random
import rclpy
from rclpy.duration import Duration
from rclpy.lifecycle import Node, State, TransitionCallbackReturn, LifecycleState
from rclpy.qos import qos_profile_sensor_data
import sys
from scipy.signal import resample
from threading import Lock
import time
import torch
import uuid

import os
spot_name = os.environ['SPOT_NAME']



class MicrophoneLifecycleNode(Node):
    def __init__(self, node_name: str):
        """
        According to nav2 documentation (https://docs.nav2.org/concepts/index.html#lifecycle-nodes-and-bond), the node's
        constructor should not contain any ROS networking setup or parameter reading; it should simply declare all the
         member variables
        :param node_name: the name of the node
        """
        # call the super node
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.get_logger().info(f'Node is in state \"unconfigured\". ')
        self.initial_unconfigured_state = True

        # declare the microphone
        self.microphone = None
        self.audio_classification_strategy = None

        # create all timer objects
        self.process_audio_buffer_timer = None

        # stores the 16-PCM formatted, unprocessed audio data coming from the main channel of microphone
        self.audio_buffer = []

        # stores 16-PCM formatted audio unprocessed audio containing speech
        self.pre_speech_buffer = None

        # stores 16-PCM formatted audio processed audio containing speech
        self.speech_buffer = None

        # number of bytes of 0s to be placed at front of audio buffer (helps with audio transcription)
        self.audio_prefetch_bytes = None

        # array of nulled bytes to be placed at front of audio buffer
        self.audio_prefetch_buffer = None

        # service indicating when to stop listening to the mic; called when we play sounds out thru speaker
        self.stop_listening_service = None
        self.stop_listening_start_time = None
        self.stop_listening_stop_time = None

        # time stamps of the first and last element of the buffer
        self.first_time_stamp_ = self.get_clock().now()
        self.last_time_stamp_ = self.get_clock().now()
        self.time_stamp_mutex_ = Lock()
        
        # time stamps we last heard speech
        self.first_heard_speech_stamp = None
        self.last_heard_speech_stamp = None
        
        # whether we sent non-verbal vocalization observation in the last cycle
        self.sent_non_verbal_last_time = False
        
        # initialize speech-to-text models
        self.model_size = 'large-v3'
        self.language = 'en'
        self.task = 'transcribe'
        self.initial_prompt = None
        self.vad_parameters = None
        self.use_vad = True
        self.device = 'cuda'
        
        # declare the whisper model
        self.transcriber = None

        # to know whether we are assessing and should process audio
        self.sub_spot_status = self.create_subscription(SpotStatus, 'spot_status', self.spot_status_callback, qos_profile_sensor_data)
        self.assessing = False

        # configure
        self._declare_parameters()
        self._create_publishers()
        self._create_timers()
        self._create_services()

        # reset the main buffer so we don't process audio from previous time this node ran
        self._reset_audio_data()

        # initialize these start/stop listening stamps
        self.stop_listening_start_time = self.get_clock().now()
        self.stop_listening_stop_time = self.get_clock().now()

        # try connecting to the audio device, if we fail, then return FAILURE, otherwise continue
        self.microphone = MicrophoneDevice(self.on_received_audio, self.get_logger(), self.get_parameter('microphone_sampling_freq_hz').value)
        self.microphone.create_pyaudio()
        if not self.microphone.find_device(self.get_parameter('microphone_name').value):
            self.get_logger().fatal("Failed to find audio device with matching name.")
            while True:
                time.sleep(1)

        self.get_logger().info(f"Found microphone at device index \"{self.microphone.device_index}\". Creating stream.")

        # try loading classifier model, if we fail, then return FAILURE, otherwise continue
        self.audio_classification_strategy = MaxArgStrategy(self.get_parameter('path_to_classifier').value, self.get_logger())
        if not self.audio_classification_strategy.load_classifier():
            self.get_logger().error("Failed to load classifiers. Not transitioning to \"active\".")
            self.stop_listening_start_time = None
            self.stop_listening_stop_time = None
            self.microphone.destroy_pyaudio()
            # loop indefinitely
            while True:
                time.sleep(1)



        # try loading whisper model, if we fail, then return FAILURE, otherwise continue
        self.vad_parameters = {
            'threshold': 0.2,  # lower bound: 0.2, sweet spot: 0.3, upper bound: 0.4
            'min_speech_duration_ms': int(1000.0 * self.get_parameter('speech_min_duration').value),
            'max_speech_duration_s': self.get_parameter('speech_max_duration').value,
            'min_silence_duration_ms': int(1000.0 * self.get_parameter('speech_continuation').value),
            'window_size_samples': 1536
        }
        if os.path.isdir(self.get_parameter('path_to_whisper').value):
            self.get_logger().info(f"Loading Whisper model from disk.")
            self.transcriber = WhisperModel(self.get_parameter('path_to_whisper').value, device=self.device, compute_type='int8', local_files_only=False)
        else:
            self.get_logger().info(f"Couldn't find Whisper model on disk. Downloading it from the internet.")
            self.transcriber = WhisperModel(self.model_size, device=self.device, compute_type='int8', local_files_only=False)


        # initialize audio prefetch buffer
        self.audio_prefetch_bytes = int(self.get_parameter('speech_prefetch').value * self.microphone.rate * self.microphone.bitdepth / 8.0)
        self.audio_prefetch_buffer = np.zeros(self.audio_prefetch_bytes, dtype=np.uint8)

        # start streaming audio data immediately
        self.microphone.create_stream()
        self.get_logger().info(f"Created microphone stream.")
        self.microphone.start_stream()
        self.get_logger().info(f"Started microphone stream.")
        self.get_logger().info(f"Node \"{self.get_name()}\" transitioned to \"activate\".")

    def spot_status_callback(self, msg: SpotStatus):
        self.assessing = msg.state == SpotStatus.ASSESSING

    def _reset_audio_data(self):
        self.audio_buffer = []

    def _declare_parameters(self) -> None:
        self.get_logger().info("Declaring node parameters.")
        self.declare_parameter('microphone_name', 'RØDE')
        self.declare_parameter('microphone_sampling_freq_hz', 48000)  # 48000 for rode and 16000 for respeaker
        self.declare_parameter('main_channel', 0)
        self.declare_parameter('speech_prefetch', 0.5)
        self.declare_parameter('update_period_s', 0.75)  # supposed to be 0.75 seconds
        self.declare_parameter('speech_continuation', 0.3)  # lower: 0.5 upper: 1.0
        self.declare_parameter('speech_max_duration', 8.0)
        self.declare_parameter('speech_min_duration', 0.25)  # upper 2.5
        self.declare_parameter('save_wav_files', False)
        self.declare_parameter('path_to_whisper', '/home/cdcl/cdcl_ws/models/whisper/')
        self.declare_parameter('path_to_classifier', "/home/cdcl/cdcl_ws/models/alertness_verbal_classifiers/")
        self.get_logger().info("Declared node parameters.")

    def _undeclare_parameter(self, name):
        if self.has_parameter(name):
            self.undeclare_parameter(name)

    def _undeclare_parameters(self) -> None:
        self.get_logger().info("Undeclaring node parameters.")
        self._undeclare_parameter('microphone_name')
        self._undeclare_parameter('microphone_sampling_freq_hz')
        self._undeclare_parameter('main_channel')
        self._undeclare_parameter('speech_prefetch')
        self._undeclare_parameter('update_period_s')
        self._undeclare_parameter('speech_continuation')
        self._undeclare_parameter('speech_max_duration')
        self._undeclare_parameter('speech_min_duration')
        self._undeclare_parameter('save_wav_files')
        self._undeclare_parameter('path_to_whisper')
        self._undeclare_parameter('path_to_classifier')
        self.get_logger().info("Undeclared node parameters.")

    def _create_publishers(self) -> None:
        """
        declare all the node's parameters
        :return:
        """
        self.get_logger().info("Creating publishers.")

        # publisher for publishing raw audio heard by the microphone
        self.pub_raw_audio = self.create_publisher(AudioData, 'raw_audio', 10)
        self.pub_speech = self.create_publisher(ObservationDataSource, 'speech', 10)
        self.pub_observation_data_source = self.create_publisher(ObservationDataSource, 'observation_data_sources', 10)
        self.pub_observation = self.create_publisher(Observation, 'observation_no_id', 10)

        self.get_logger().info("Created publishers..")

    def _destroy_publishers(self) -> None:
        self.get_logger().info("Destroying publishers.")

        # publisher for publishing raw audio heard by the microphone
        self.destroy_publisher('raw_audio')
        self.pub_raw_audio = None
        self.destroy_publisher('speech')
        self.pub_speech = None
        self.destroy_publisher('observation_no_id')
        self.pub_observation = None

        self.get_logger().info("Destroyed publishers.")

    def _create_services(self) -> None:
        self.get_logger().info("creating services")
        self.stop_listening_service = self.create_service(StopListening, 'stop_listening', self.stop_listening_callback)
        self.get_logger().info("created services")

    def _destroy_services(self) -> None:
        self.get_logger().info("destroying services")
        self.destroy_service('stop_listening')
        self.stop_listening_service = None
        self.get_logger().info("destroyed services")

    def _create_timers(self) -> None:
        if self.process_audio_buffer_timer is None:
            self.process_audio_buffer_timer = self.create_timer(self.get_parameter('update_period_s').value, self.on_process_audio_buffer_callback)

    def _destroy_timers(self) -> None:
        if self.process_audio_buffer_timer is not None:
            self.destroy_timer(self.process_audio_buffer_timer)
            self.process_audio_buffer_timer = None

    def filter_audio(self, buffered_audio):
        """
        filters noise and then amplifies audio buffer
        :param buffered_audio:
        :return:
        """
        # filter the raw audio data
        filtered_buffered_audio = nr.reduce_noise(y=buffered_audio, sr=self.microphone.rate, stationary=False)

        # amplify the filtered audio data to be 80% peak-to-peak the dynamic range of the speaker
        max_amplitude = max(np.abs(np.max(filtered_buffered_audio)), np.abs(np.min(filtered_buffered_audio)))
        amplified_filtered_buffered_audio = np.round(0.80 * (2**15-1) / (max_amplitude) * filtered_buffered_audio).astype(np.int16)
        return amplified_filtered_buffered_audio

    def on_process_audio_buffer_callback(self):
        # if the microphone or classification strategy isn't initialized, just return
        if self.microphone is None or self.audio_classification_strategy is None or self.audio_prefetch_bytes is None or self.audio_prefetch_buffer is None:
            return

        # copy timestamps and audio buffer into local variables, so the other thread(s) may update them some more
        last_time_stamp = None
        first_time_stamp = None
        with self.time_stamp_mutex_:
            last_time_stamp = self.last_time_stamp_
            first_time_stamp = self.first_time_stamp_
        buffered_audio = self.audio_buffer
        self.audio_buffer = []

        if not self.assessing:
            # clear all speech buffers
            self.speech_buffer = None
            self.last_heard_speech_stamp = None
            self.first_heard_speech_stamp = None
            return

        # if the buffer is empty, just return
        if len(buffered_audio) == 0:
            return

        # append amplified and filtered audio buffer to array
        if self.pre_speech_buffer is None:
            self.pre_speech_buffer = [[self.audio_prefetch_buffer], buffered_audio]
            self.get_logger().debug("Received first audio message. Creating pre-speech buffer.")
            return
        else:
            assert len(self.pre_speech_buffer) == 2 or len(self.pre_speech_buffer) == 3
            self.pre_speech_buffer = [[self.audio_prefetch_buffer], self.pre_speech_buffer[-1], buffered_audio]
            self.get_logger().debug("Appending audio to pre-speech buffer.")
            assert len(self.pre_speech_buffer) == 3

        self.get_logger().info(f'Reaching this point')

        # filter and amplify the audio
        raw_audio = np.hstack(buffered_audio).astype(np.int8)
        # amplified_filtered_audio = raw_audio
        amplified_filtered_audio = self.filter_audio(np.hstack([elem for l in self.pre_speech_buffer for elem in l]))

        # hash the audio data to create a unique id for messages down the line...
        unique_id = random.randint(-2**31, 2**31 - 1)

        # classify the most recent audio snippet we got
        target_sample_rate = 16000
        if self.microphone.rate != target_sample_rate:
            amplified_filtered_audio_f32 = amplified_filtered_audio.astype(np.float32)
            num_samples_resampled = int(len(amplified_filtered_audio) * target_sample_rate / self.microphone.rate)
            audio_resampled = resample(amplified_filtered_audio_f32, num_samples_resampled)
            amplified_filtered_audio = np.clip(np.round(audio_resampled), -32768, 32767).astype(np.int16)
        output_tensor = self.audio_classification_strategy.classify_audio(amplified_filtered_audio)
        output = self.audio_classification_strategy.apply_strategy(output_tensor)
        if output is not None:
            respiratory_distress_outputs, verbal_alertness_outputs = output
            respiratory_distress_label = torch.argmax(respiratory_distress_outputs).item()
            verbal_alertness_label = torch.argmax(verbal_alertness_outputs).item()
            # respiratory_distress_label = max(respiratory_distress_outputs.items(), key=lambda item: item[1])[0]
            # verbal_alertness_label = max(verbal_alertness_outputs.items(), key=lambda item: item[1])[0]
            self.get_logger().info(f"Respiratory Distress: {respiratory_distress_outputs}, Verbal Alertness: {verbal_alertness_outputs}")

            # publish the verbal alertness classification, if it's non-verbal vocalization and we didn't send last time
            already_sent_observation_data_source = False
            observation_data_source = ObservationDataSource(
                data_source_id=unique_id,
                platform_name=spot_name,
                raw_audio=raw_audio
            )
            if not self.sent_non_verbal_last_time:
                if verbal_alertness_label == 1 or verbal_alertness_label == 0:
                    alertness_observation = Observation()
                    alertness_observation.platform_name = spot_name #self.get_parameter('robot_name').value
                    alertness_observation.data_source_id = unique_id
                    alertness_observation.stamp = first_time_stamp.to_msg()
                    alertness_observation.observation_module = ObservationModule.AST_ALERTNESS_VERBAL
                    alertness_observation.observation = verbal_alertness_outputs.tolist()
                    self.pub_observation.publish(alertness_observation)
                    self.sent_non_verbal_last_time = True
                    self.get_logger().debug("Detected a non-verbal vocalization. Publishing observation message.")

                    already_sent_observation_data_source = True
                    self.pub_observation_data_source.publish(observation_data_source)


                # publish the respiratory distress classification, if it's non-verbal vocalization
                if respiratory_distress_label == 1:
                    alertness_observation = Observation()
                    alertness_observation.platform_name = spot_name  #self.get_parameter('robot_name').value
                    alertness_observation.data_source_id = unique_id
                    alertness_observation.stamp = first_time_stamp.to_msg()
                    alertness_observation.observation_module = ObservationModule.AST_RESPIRATORY_DISTRESS
                    alertness_observation.observation = respiratory_distress_outputs.tolist()
                    self.pub_observation.publish(alertness_observation)
                    self.sent_non_verbal_last_time = True
                    self.get_logger().debug("Detected a respiratory distress. Publishing observation message.")

                    if not already_sent_observation_data_source:
                        self.pub_observation_data_source.publish(observation_data_source)

            else:
                self.sent_non_verbal_last_time = False

        # transcribe the most recent audio snippet
        transcript = self._transcribe_speech(amplified_filtered_audio)

        # if there was an error transcribing the speech, add the audio to the speech buffer out of an abundance of caution (maybe there is actually speech there)
        if transcript is None or output is None:
            self.get_logger().warn("Transcription of latest speech buffer failed. Appending data to speech buffer anyway.")
            if self.speech_buffer is None:
                self.speech_buffer = self.pre_speech_buffer
                self.first_heard_speech_stamp = self.get_clock().now()
            else:
                self.speech_buffer.append(buffered_audio)
            return

        self.get_logger().debug(f"Transcribed initial speech with no errors. Probability of Speech: {transcript['prob_speech']}.")

        # speech was detected
        if verbal_alertness_label == 0 and transcript['prob_speech'] >= 0.50:  # tweak these numbers?
            self.get_logger().debug(f"Received what is most likely speech. Appending audio to speech buffer.")
            self.last_heard_speech_stamp = self.get_clock().now()
            if self.speech_buffer is None:
                self.speech_buffer = self.pre_speech_buffer
                self.first_heard_speech_stamp = self.get_clock().now()
            else:
                self.speech_buffer.append(buffered_audio)
            return

        # no speech detected
        else:
            # We didn't receive any speech anyway, so just clear the speech buffer
            if self.last_heard_speech_stamp is None:
                self.get_logger().debug(f"Haven't heard speech in a while. Not appending to the buffer.")
                return
            elif self.get_clock().now() - self.last_heard_speech_stamp <= rclpy.duration.Duration(seconds=0.5):   # tweak these numbers: 1.0 works
                self.get_logger().debug(f"Haven't heard speech, but I'm waiting to see if pause in speech is because person is done talking or just temporary pause. Appending audio to speech buffer.")
                self.speech_buffer.append(buffered_audio)
                return

        time_stopped_listening = self.get_clock().now()
        raw_speech_audio = np.hstack(buffered_audio).astype(np.int8)
        amplified_filtered_audio = self.filter_audio(np.hstack([elem for l in self.speech_buffer for elem in l]))
        # amplified_filtered_audio = np.hstack([elem for l in self.speech_buffer for elem in l])

        # classify the whole speech buffer
        target_sample_rate = 16000
        if self.microphone.rate != target_sample_rate:
            amplified_filtered_audio_f32 = amplified_filtered_audio.astype(np.float32)
            num_samples_resampled = int(len(amplified_filtered_audio) * target_sample_rate / self.microphone.rate)
            audio_resampled = resample(amplified_filtered_audio_f32, num_samples_resampled)
            amplified_filtered_audio = np.clip(np.round(audio_resampled), -32768, 32767).astype(np.int16)
        output_tensor = self.audio_classification_strategy.classify_audio(amplified_filtered_audio)
        output = self.audio_classification_strategy.apply_strategy(output_tensor)

        transcript = self._transcribe_speech(amplified_filtered_audio)
        if transcript is None or output is None:
            self.get_logger().warn(f"While trying to transcribe the speech buffer, an error occurred, so we missed out on speech. Clearing buffers.")
            self.speech_buffer = None
            self.last_heard_speech_stamp = None
            self.first_heard_speech_stamp = None
            return

        respiratory_distress_outputs, verbal_alertness_outputs = output
        respiratory_distress_label = torch.argmax(respiratory_distress_outputs).item()
        verbal_alertness_label = torch.argmax(verbal_alertness_outputs).item()
        self.get_logger().info(f"Respiratory Distress: {respiratory_distress_outputs}, Verbal Alertness: {verbal_alertness_outputs}")

        # initialize a speech object
        speech = ObservationDataSource()
        speech.data_source_id = random.randint(-2**31, 2**31 - 1)
        speech.raw_audio = raw_speech_audio.tolist()  #amplified_filtered_audio.tolist()
        speech.platform_name = spot_name
        speech.audio_start = self.first_heard_speech_stamp.to_msg()
        speech.audio_end_time = self.last_heard_speech_stamp.to_msg()


        # publish the transcription of the entire speech buffer.
        if verbal_alertness_label == 0 and transcript['prob_speech'] >= 0.60:
            now = self.get_clock().now()
            self.get_logger().info(f"Detected Speech Transcription: {transcript}. Processing delay: {now - time_stopped_listening}. Stop Listening Delay: {now - self.last_heard_speech_stamp}. Publishing observations and clearing the buffers.")

            # publish speech
            speech.audio_transcript = transcript['transcript']
            speech.audio_transcript = transcript['transcript']
            self.pub_speech.publish(speech)
            self.pub_observation_data_source.publish(speech)

            # publish an observation corresponding to speech
            alertness_observation = Observation()
            alertness_observation.platform_name = spot_name  #self.get_parameter('robot_name').value
            alertness_observation.data_source_id = unique_id
            alertness_observation.stamp = self.first_heard_speech_stamp.to_msg()
            alertness_observation.observation_module = ObservationModule.WHISPER_ALERTNESS_VERBAL
            alertness_observation.observation = [transcript['prob_speech'], 0.0, 1.0 - transcript['prob_speech']]  # todo, this is a speech classifier, so it detects presence or absence of NORMAL.
            # self.pub_observation.publish(alertness_observation)


        # clearing the buffers
        self.speech_buffer = None
        self.last_heard_speech_stamp = None
        self.first_heard_speech_stamp = None

    def on_received_audio(self, data, channel) -> None:
        """
        adds most recent audio data from microphone to a buffer to be processed later
        :param data: new microphone data
        :param channel: the channel providing the data
        :return: null
        """
        if channel == self.get_parameter('main_channel').value:
            # immediately publish all unprocessed audio --- might contain spot speech but that's good!
            arr_int16 = np.array(data, dtype=np.int16)
            # self.get_logger().info(f'data = {data}')
            self.pub_raw_audio.publish(AudioData(data=arr_int16.view(np.uint8).tolist()))
            # self.pub_raw_audio.publish(AudioData(data=data.astype(np.uint8)))

            # compute duration of audio segment we just received
            seconds_ago = len(data) / (self.microphone.rate * self.microphone.bitdepth / 8.0)

            # # ignore and flush the audio buffer if we received audio during period that we were playing sound
            # if self.stop_listening_start_time <= self.get_clock().now() <= self.stop_listening_stop_time:
            #     self.audio_buffer = []
            #     self.get_logger().info("Ignoring audio")
            #     return
            #
            # # update the first and last times we received audio data
            # if len(self.audio_buffer) == 0:
            #     with self.time_stamp_mutex_:
            #         self.first_time_stamp_ = self.get_clock().now() - Duration(seconds=seconds_ago)
            #         self.last_time_stamp_ = self.get_clock().now()
            # else:
            #     with self.time_stamp_mutex_:
            #         self.last_time_stamp_ = self.get_clock().now()
            #
            # # append the most recent data onto the end of the buffer
            # self.audio_buffer.append(data)

    def stop_listening_callback(self, request, response):
        """
        assigns values "self.stop_listening_start_time" and "self.stop_listening_stop_time", which the node will use
        to ignore any audio heard within those two time stamps
        :param request: the ros2 request
        :param response: the response
        :return:
        """
        self.get_logger().info(f"Processing request to stop listening.")
        self.stop_listening_start_time = rclpy.time.Time().from_msg(request.stop_listen_time)
        self.stop_listening_stop_time = rclpy.time.Time().from_msg(request.start_listen_time)
        response.success = True
        self.get_logger().info(f"Processed request to stop listening. Stop Listening Starting: {self.stop_listening_start_time}, Stop Listening Ending: {self.stop_listening_stop_time}")
        return response

    def _transcribe_speech(self, audio_data: npt.NDArray) -> dict[str, int | None] | None:
        audio_data_f32 = pcm_to_f32(audio_data)
        result, info = self.transcriber.transcribe(
            audio_data_f32,
            initial_prompt=self.initial_prompt,
            language=self.language,
            task=self.task,
            # vad_filter=False,
            # vad_parameters=None
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None
        )

        if result:
            return {'transcript': result[0].text, 'prob_speech': 1.0 - result[0].no_speech_prob }
        return None


def main(args=None) -> None:
    rclpy.init(args=args)
    microphone_node = MicrophoneLifecycleNode('microphone_lifecycle_node')
    rclpy.spin(microphone_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
