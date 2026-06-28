#!/usr/bin/env python3

# from audio_common_msgs.msg import AudioData, AudioDataStamped
# from cdcl_umd_msgs.srv import PlaySound
# from cdcl_umd_msgs.srv import StopListening
# from std_msgs.msg import Empty
# import os
# import rclpy
# from rclpy.node import Node
# from rclpy.duration import Duration
# from speaker.speaker_device import USBSpeakerDevice, JackSpeakerDevice
# from std_msgs.msg import Header
# import wave
# import numpy as np
# import soundfile as sf

# # Import Kokoro
# from kokoro import KPipeline


# class SpeakerNode(Node):
#     def __init__(self):
#         """
#         initialize the node as a service server
#         """
#         super().__init__('speaker_node')

#         # the name of the speech service.
#         self.speech_service_name_ = self.declare_parameter('speech_service_name', 'speak')
#         self.stop_listening_service_name_ = self.declare_parameter('stop_listening_service_name', 'stop_listening')

#         # Output file location (kept parameter name 'xtts_output_file' to prevent breaking external launch files)
#         self.output_file_ = self.declare_parameter('xtts_output_file', '/home/cdcl/cdcl_ws/src/spot_audio/data/output.wav')

#         # create a service server called 'speak' that accepts PlaySound requests
#         self.play_sound_srv_ = self.create_service(PlaySound, self.speech_service_name_.value, self.play_sound_callback)

#         # create a client for the stop listening service
#         self.stop_listening_client_ = self.create_client(StopListening, self.stop_listening_service_name_.value)

#         # publish to a heartbeat topic every once in awhile
#         self.empty_pub_ = self.create_publisher(
#             Empty,
#             'speaker/heartbeat',
#             10
#         )
#         self.spot_voice_ = self.create_publisher(
#             AudioDataStamped,
#             'speaker/voice',
#             10
#         )
#         self.heartbeat_timer_ = self.create_timer(2.5, self.heartbeat_callback)

#         # initialize the speaker device
#         self.speaker_device_ = JackSpeakerDevice('alsa_output.pci-0000_00_1f.3.analog-stereo')
#         self.speaker_device_.set_device('alsa_output.pci-0000_00_1f.3.analog-stereo')

#         # for testing only
#         #self.speaker_device_ = JackSpeakerDevice('alsa_output.usb-Generic_USB2.0_Device_20121120222016-00.analog-stereo')
#         #self.speaker_device_.set_device('alsa_output.usb-Generic_USB2.0_Device_20121120222016-00.analog-stereo')

#         # initialize the Kokoro model
#         self.get_logger().info(f"Loading Kokoro Model...")
#         # 'a' = American English. Model weights will auto-download on first run.
#         self.tts_pipeline_ = KPipeline(lang_code='a') 
#         self.kokoro_voice_ = 'am_adam'

#         # generate the first speech, just so the model is warmed up
#         self._run_tts('Hello world! This is a test!')
#         self._run_tts('I try to generate at least three sounds first')
#         self._run_tts('Spot is ready to go!')
#         self.seq = 0

#     def heartbeat_callback(self) -> None:
#         self.empty_pub_.publish(Empty())

#     def _run_tts(self, text: str) -> None:
#         """
#         generates a .wav file from text
#         :param text: the text whose audio we are generating
#         :return: void
#         """
        
#         # Kokoro processes text and yields chunks of audio
#         generator = self.tts_pipeline_(text, voice=self.kokoro_voice_, speed=0.85)
        
#         all_audio = []
#         for gs, ps, audio in generator:
#             if audio is not None:
#                 all_audio.append(audio)
        
#         if len(all_audio) > 0:
#             # Concatenate chunks and save to file at Kokoro's native 24kHz
#             combined_audio = np.concatenate(all_audio)
#             sf.write(self.output_file_.value, combined_audio, 24000)
#         else:
#             self.get_logger().error(f"Kokoro failed to generate audio for text: {text}")
#             # Generate a tiny silent file as a fallback to prevent duration/playback crashes
#             sf.write(self.output_file_.value, np.zeros(24000), 24000)

#     def _compute_wavfile_duration(self, path) -> float:
#         """
#         computes the duration of a wav file
#         :param path: the path to the wav file
#         :return: its duration in seconds as a float
#         """
#         wav = wave.open(path)
#         frame_count = wav.getnframes()
#         frame_rate = wav.getframerate()
#         wav.close()
#         return frame_count / float(frame_rate)

#     def play_sound_callback(self, request, response):
#         """
#         plays a sound through the speaker device, if the file exists
#         :param request: the playsound request, containing the sound file to be played
#         :param response: the playsound response, containing whether the sound was played successfully
#         :return:
#         """

#         # remove the old output file if it exists
#         if os.path.exists(self.output_file_.value):
#             os.remove(self.output_file_.value)

#         # generate the .wav file
#         self.get_logger().info("Received play sound request \"{}\"".format(request.text.lower()))
#         self._run_tts(request.text.lower())

#         # check that file exists
#         if not os.path.exists(self.output_file_.value) or not os.path.isfile(self.output_file_.value):
#             response.success = False
#             return response

#         # ask the microphone to stop listening for the next <duration of wavfile> seconds
#         duration_s = self._compute_wavfile_duration(self.output_file_.value)

#         # try to recreate the audio, if it's empty
#         if duration_s == 0.0:
#             self.get_logger().error(f'Generated empty audio file! Retrying...')
#             self._run_tts(request.text.lower())
#             duration_s = self._compute_wavfile_duration(self.output_file_.value)

#         self.get_logger().info(f"Audio Duration: {duration_s:.2f} [s]")

#         stop_listening_request = StopListening.Request()
#         right_now = self.get_clock().now()
#         stop_listening_request.stop_listen_time = right_now.to_msg()
#         stop_listening_request.start_listen_time = (right_now + Duration(seconds=duration_s + 0.5)).to_msg()
        
#         # call async without waiting
#         self.stop_listening_client_.call_async(stop_listening_request)  

#         # play the sound through the speaker
#         response.start_time = self.get_clock().now().to_msg()
#         self.speaker_device_.play_sound(self.output_file_.value)
#         response.end_time = self.get_clock().now().to_msg()

#         self.spot_voice_.publish(
#             AudioDataStamped(
#                 audio=SpeakerNode.wav_to_audio_data(self.output_file_.value),
#                 header=Header(
#                     stamp=response.start_time
#                 )
#             )
#         )
#         self.seq += 1

#         # once the speaker stops playing the sound, report success to the user
#         response.success = True
#         return response

#     @staticmethod
#     def wav_to_audio_data(path: str) -> AudioData:
#         """
#         reads a .wav file and returns an AudioData message containing the raw audio
#         :param path: path to the .wav file
#         :return: AudioData: ROS message containing raw audio data
#         """
#         with wave.open(path, 'rb') as wav_file:
#             raw_bytes = wav_file.readframes(wav_file.getnframes())
#         return AudioData(data=bytearray(raw_bytes))


# def main(args=None):
#     rclpy.init(args=args)
#     speaker_node = SpeakerNode()
#     try:
#         rclpy.spin(speaker_node)
#     except KeyboardInterrupt:
#         pass
#     speaker_node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3

from audio_common_msgs.msg import AudioData, AudioDataStamped
from cdcl_umd_msgs.srv import PlaySound
from cdcl_umd_msgs.srv import StopListening
from std_msgs.msg import Empty
import os
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from speaker.speaker_device import USBSpeakerDevice, JackSpeakerDevice, PulseSpeakerDevice
from std_msgs.msg import Header
import wave
import numpy as np
import soundfile as sf

# Import Kokoro
from kokoro import KPipeline


class SpeakerNode(Node):
    def __init__(self):
        """
        initialize the node as a service server
        """
        super().__init__('speaker_node')

        # the name of the speech service.
        self.speech_service_name_ = self.declare_parameter('speech_service_name', 'speak')
        self.stop_listening_service_name_ = self.declare_parameter('stop_listening_service_name', 'stop_listening')
        
        # Target native PulseAudio sink
        self.speaker_device_name_ = self.declare_parameter('speaker_device_name', 'robot_aec_speaker')

        # Output file location
        self.output_file_ = self.declare_parameter('xtts_output_file', '/home/cdcl/cdcl_ws/src/spot_audio/data/output.wav')

        # create a service server called 'speak' that accepts PlaySound requests
        self.play_sound_srv_ = self.create_service(PlaySound, self.speech_service_name_.value, self.play_sound_callback)

        # create a client for the stop listening service
        self.stop_listening_client_ = self.create_client(StopListening, self.stop_listening_service_name_.value)

        # publish to a heartbeat topic every once in awhile
        self.empty_pub_ = self.create_publisher(Empty, 'speaker/heartbeat', 10)
        self.spot_voice_ = self.create_publisher(AudioDataStamped, 'speaker/voice', 10)
        
        self.heartbeat_timer_ = self.create_timer(2.5, self.heartbeat_callback)

        # initialize the native PulseAudio speaker device
        self.speaker_device_ = PulseSpeakerDevice(self.speaker_device_name_.value)

        # initialize the Kokoro model
        self.get_logger().info(f"Loading Kokoro Model...")
        # 'a' = American English. Model weights will auto-download on first run.
        self.tts_pipeline_ = KPipeline(lang_code='a') 
        self.kokoro_voice_ = 'am_adam'

        # generate the first speech, just so the model is warmed up
        self._run_tts('Hello world! This is a test!')
        self._run_tts('I try to generate at least three sounds first')
        self._run_tts('Spot is ready to go!')
        self.seq = 0

    def heartbeat_callback(self) -> None:
        self.empty_pub_.publish(Empty())

    def _run_tts(self, text: str) -> None:
        """
        generates a .wav file from text
        :param text: the text whose audio we are generating
        :return: void
        """
        # Kokoro processes text and yields chunks of audio
        generator = self.tts_pipeline_(text, voice=self.kokoro_voice_, speed=0.85)
        
        all_audio = []
        for gs, ps, audio in generator:
            if audio is not None:
                all_audio.append(audio)
        
        if len(all_audio) > 0:
            # Concatenate chunks and save to file at Kokoro's native 24kHz
            combined_audio = np.concatenate(all_audio)
            sf.write(self.output_file_.value, combined_audio, 24000)
        else:
            self.get_logger().error(f"Kokoro failed to generate audio for text: {text}")
            # Generate a tiny silent file as a fallback to prevent duration/playback crashes
            sf.write(self.output_file_.value, np.zeros(24000), 24000)

    def _compute_wavfile_duration(self, path) -> float:
        """
        computes the duration of a wav file
        :param path: the path to the wav file
        :return: its duration in seconds as a float
        """
        wav = wave.open(path)
        frame_count = wav.getnframes()
        frame_rate = wav.getframerate()
        wav.close()
        return frame_count / float(frame_rate)

    def play_sound_callback(self, request, response):
        """
        plays a sound through the speaker device, if the file exists
        :param request: the playsound request, containing the sound file to be played
        :param response: the playsound response, containing whether the sound was played successfully
        :return:
        """
        # remove the old output file if it exists
        if os.path.exists(self.output_file_.value):
            os.remove(self.output_file_.value)

        # generate the .wav file
        self.get_logger().info(f"Received play sound request \"{request.text.lower()}\"")
        self._run_tts(request.text.lower())

        # check that file exists
        if not os.path.exists(self.output_file_.value) or not os.path.isfile(self.output_file_.value):
            response.success = False
            return response

        # ask the microphone to stop listening for the next <duration of wavfile> seconds
        duration_s = self._compute_wavfile_duration(self.output_file_.value)

        # try to recreate the audio, if it's empty
        if duration_s == 0.0:
            self.get_logger().error(f'Generated empty audio file! Retrying...')
            self._run_tts(request.text.lower())
            duration_s = self._compute_wavfile_duration(self.output_file_.value)

        self.get_logger().info(f"Audio Duration: {duration_s:.2f} [s]")

        stop_listening_request = StopListening.Request()
        right_now = self.get_clock().now()
        stop_listening_request.stop_listen_time = right_now.to_msg()
        stop_listening_request.start_listen_time = (right_now + Duration(seconds=duration_s + 0.5)).to_msg()
        
        # call async without waiting
        self.stop_listening_client_.call_async(stop_listening_request)  

        # play the sound directly through PulseAudio via our new device
        response.start_time = self.get_clock().now().to_msg()
        self.speaker_device_.play_sound(self.output_file_.value)
        response.end_time = self.get_clock().now().to_msg()

        self.spot_voice_.publish(
            AudioDataStamped(
                audio=SpeakerNode.wav_to_audio_data(self.output_file_.value),
                header=Header(
                    stamp=response.start_time
                )
            )
        )
        self.seq += 1

        # once the speaker stops playing the sound, report success to the user
        response.success = True
        return response

    @staticmethod
    def wav_to_audio_data(path: str) -> AudioData:
        """
        reads a .wav file and returns an AudioData message containing the raw audio
        :param path: path to the .wav file
        :return: AudioData: ROS message containing raw audio data
        """
        with wave.open(path, 'rb') as wav_file:
            raw_bytes = wav_file.readframes(wav_file.getnframes())
        return AudioData(data=bytearray(raw_bytes))


def main(args=None):
    rclpy.init(args=args)
    speaker_node = SpeakerNode()
    try:
        rclpy.spin(speaker_node)
    except KeyboardInterrupt:
        pass
    speaker_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()