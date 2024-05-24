#!/usr/bin/env python3

# standard ros2 imports
import rclpy
from rclpy.node import Node
import rclpy.time
import rclpy.duration
import rclpy.timer

# for outputting raw audio data
from audio_common_msgs.msg import AudioData

# for getting status of leds on microphone
from std_msgs.msg import ColorRGBA

# for outputting speech transcripts and speech activity
from cdcl_umd_msgs.msg import Speech
from cdcl_umd_msgs.msg import SpeechActivity

# python api for communicating with the respeaker over ros2
from spot_audio.interface import RespeakerInterface
from spot_audio.audio import RespeakerAudio

# python wrapper for whisper speech-to-text model
from spot_audio.transcriber import WhisperModel

# math stuff required for manipulating bytes and computing DOA
import math
import numpy as np
import angles
import tf_transformations as T


class RespeakerNode(Node):
    def __init__(self):
        super().__init__('respeaker_node')
        # new strategy: make speech_min_duration lower or vad threshold lower... this will increase frequency of false positives
        # then increase whisper vad

        # the sensor frame of the microphone, needed for interpreting the DOA measurements
        self.sensor_frame_id = self.declare_parameter('sensor_frame_id', 'respeaker_base')

        # TODO not sure what this is
        self.speech_prefetch = self.declare_parameter('speech_prefetch', 0.5)

        # the frequency with which we will 
        # self.update_period_s = self.declare_parameter('update_period_s', 0.1)
        self.update_period_s = self.declare_parameter('update_period_s', 1.0)

        # the speaker has 4 channels used for detecting DOA, we choose one of them as the "main"
        # channel and get audio from that channel specifically, all channels are the same
        self.main_channel = self.declare_parameter('main_channel', 0)

        # the minimum amount of time a person can take a break between words before we finish with the
        # transcript and send it to the user
        self.speech_continuation = self.declare_parameter('speech_continuation', 0.3)  # lower: 0.5 upper: 1.0 

        # the maximum duration a chunk of speech can be before we try to transcribe it
        self.speech_max_duration = self.declare_parameter('speech_max_duration', 8.0)

        # the minimum duration of a chunk of speech before we reject it as nothing
        self.speech_min_duration = self.declare_parameter('speech_min_duration', 0.25)  # upper 2.5

        # offsets DOA measurements by fixed amount (interesting if you mount the respeaker on Spot in a weird way)
        self.doa_yaw_offset = self.declare_parameter('doa_yaw_offset', 90.0)

        # the period that we will rerun the timer callback, which manages how often we try to detect speech
        self.update_period_s_ = 0.1

        # python objects for getting info from the respeaker
        self.respeaker = RespeakerInterface()
        self.respeaker_audio = RespeakerAudio(self.on_audio, suppress_error=True)  # on audio is a callback function
        self.respeaker.set_vad_threshold(2.0)  # lower bound: 1.9, sweet spot: 1.75 (could make continuation shorter, and this might be fine) upper bound: 2.5

        # contains all audio datas containing speech in 16-PCM format encoded as bytes 
        self.speech_audio_buffer = []

        # whether we think we can hear a person speaking
        self.is_speaking = True  # was False originall

        # a timestamp containing the last time we heard someone say a word
        self.speech_stopped = self.get_clock().now()

        # TODO: NOT SURE
        self.prev_is_voice = None
        self.prev_doa = None

        # publishes raw audio data
        self._pub_audio = self.create_publisher(AudioData, 'audio', 10)

        # publishes "Speech" datatype, which contains audio of the person talking, a transcript of the speech, and the DOA
        self._pub_speech = self.create_publisher(Speech, 'speech_audio', 10)

        # publishes whether a person is actively speaking
        self._pub_speech_activity = self.create_publisher(SpeechActivity, 'triage/speech_activity', 10)

        # a timer callback to be called frequently enough to determine if someone is actively speaking
        self._timer = self.create_timer(self.update_period_s.value, self.on_timer)

        # TODO: not sure
        self.speech_prefetch_bytes = int(
            self.speech_prefetch.value * self.respeaker_audio.rate * self.respeaker_audio.bitdepth / 8.0)
        self.speech_prefetch_buffer = np.zeros(self.speech_prefetch_bytes, dtype=np.uint8)

        # start collecting audio data
        self.respeaker_audio.start()
 
        # for changing the LEDs and reading the led status from respeaker_ros
        self.timer_led = None
        self.sub_led = self.create_subscription(ColorRGBA, "status_led", self.on_status_led, 1)

        # speech-to-text stuff
        self.model_sizes = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v2", "large-v3",
        ]
        # self.model_size = 'large-v2'  # absolutely sucks
        self.model_size = 'large-v3'  # absolutely sucks
        self.language = 'en'
        self.task = 'transcribe'
        self.initial_prompt = None
        self.get_logger().info('self.speech_prefetch_bytes: {0}'.format(self.speech_prefetch_bytes))
        self.vad_parameters={
            'threshold': 0.3,  # lower bound: 0.2, sweet spot: 0.3, upper bound: 0.4
            'min_speech_duration_ms': int(1000.0 * self.speech_min_duration.value),
            'max_speech_duration_s': self.speech_max_duration.value,
            'min_silence_duration_ms': int(1000.0 * self.speech_continuation.value),
            'window_size_samples': 1536
        }
        self.use_vad=True
        self.device = 'cuda'

        self.transcriber = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type='int8',
            local_files_only=False
        )

    def on_audio(self, data, channel):
        if channel == self.main_channel.value:
            self._pub_audio.publish(AudioData(data=data.astype(np.uint8)))
            if len(self.speech_audio_buffer) == 0:
                self.speech_audio_buffer = [self.speech_prefetch_buffer]
            self.speech_audio_buffer.append(data)

    def on_timer(self):
        stamp = self.get_clock().now()

        # compute the direction of arrival
        doa_rad = math.radians(self.respeaker.direction - 180.0)
        doa_rad = angles.shortest_angular_distance(doa_rad, math.radians(self.doa_yaw_offset.value))

        # save the audio buffer into another buffer and clear the original buffer
        buffered_speech = self.speech_audio_buffer
        self.speech_audio_buffer = []
        if len(buffered_speech) == 0:
            return
        buffered_speech = np.hstack(buffered_speech)

        # self.get_logger().info("VAD Threshold {0} VAD {1} MicSpeech {2}, ROS Speech {3}".format(self.respeaker.get_vad_threshold(), self.respeaker.is_voice(), self.respeaker.is_speech(), self.is_speaking))

        # transcribe the speech and put in message
        transcript = self.transcribe_speech(buffered_speech).strip()

        # determine if we detected speech
        # TODO: IGNORE "Thank you"
        speech_detected = int(transcript != '' and 'Thank' not in transcript)

        # publish whether speech was detected
        speech_activity = SpeechActivity()
        speech_activity.stamp = stamp.to_msg()
        speech_activity.observation = speech_detected
        self._pub_speech_activity.publish(speech_activity)

        # check if there was any speech by seeing if there is anything in the transcript
        if speech_detected:
            # prepare to send the speech message
            speech_msg = Speech()
            speech_msg.header.frame_id = self.sensor_frame_id.value
            speech_msg.header.stamp = stamp.to_msg()

            # put raw speech data into message
            speech_msg.raw_audio = buffered_speech.tolist()

            # print the transcript for convenience
            self.get_logger().info("Transcript {0}".format(transcript))

            # use direction-of-arrival estimate to compute quaternion of speaker in microphone frame
            orientation = T.quaternion_from_euler(doa_rad, 0, 0)
            speech_msg.doa.position.x = 0.0
            speech_msg.doa.position.y = 0.0
            speech_msg.doa.orientation.w = orientation[0]
            speech_msg.doa.orientation.x = orientation[1]
            speech_msg.doa.orientation.y = orientation[2]
            speech_msg.doa.orientation.z = orientation[3]

            # save the transcript
            speech_msg.transcript = transcript

            # publish the speech message
            self._pub_speech.publish(speech_msg)


    def on_status_led(self, msg):
        """
        sets the RGB of the respeaker depending on what the user requested
        """
        self.respeaker.set_led_color(r=msg.r, g=msg.g, b=msg.b, a=msg.a)
        if self.timer_led and self.timer_led.is_alive():
            self.timer_led.destroy()
        self.timer_led = rclpy.timer.Timer(rclpy.duration.Duration(3.0),
                                     lambda e: self.respeaker.set_led_trace(),
                                     oneshot=True)
    
    def transcribe_speech(self, buffered_speech_int16):
        """
        calls the whisper model on the speech data
        """
        # note you cannot send np.uint8, you must send a float32 as per whisper
        buffered_speech_float32 = self.bytes_to_float_array(buffered_speech_int16)
        result, info = self.transcriber.transcribe(
            buffered_speech_float32,
            initial_prompt=self.initial_prompt,
            language=self.language,
            task=self.task,
            # vad_filter=False,
            # vad_parameters=None
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None
        )

        if result:
            return result[0].text
        return ""

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
    mic_node = RespeakerNode()

    # if we receive a keyboard interupt, then quit gracefully
    try:
        rclpy.spin(mic_node)
    except KeyboardInterrupt:
        pass

    mic_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
