#!/usr/bin/env python3


from audio_common_msgs.msg import AudioData, AudioDataStamped
from cdcl_umd_msgs.srv import PlaySound
from cdcl_umd_msgs.srv import StopListening
from std_msgs.msg import Empty
import os
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import torch
from speaker.speaker_device import USBSpeakerDevice, JackSpeakerDevice
from std_msgs.msg import Header
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torchaudio
import wave


class SpeakerNode(Node):
    def __init__(self):
        """
        initialize the node as a service server
        """
        super().__init__('speaker_node')

        # the name of the speech service.
        self.speech_service_name_ = self.declare_parameter('speech_service_name', 'speak')
        self.stop_listening_service_name_ = self.declare_parameter('stop_listening_service_name', 'stop_listening')

        # xtts related configuration files
        self.xtts_config_file_ = self.declare_parameter('xtts_config_file', '/home/cdcl/cdcl_ws/models/xtts/config.json')
        self.xtts_speaker_audio_file_ = self.declare_parameter('xtts_speaker_audio_file', '/home/cdcl/cdcl_ws/models/xtts/test.wav')
        self.xtts_output_file_ = self.declare_parameter('xtts_output_file', '/home/cdcl/cdcl_ws/src/spot_audio/data/output.wav')

        # create a service server called 'speak' that accepts PlaySound requests
        self.play_sound_srv_ = self.create_service(PlaySound, self.speech_service_name_.value, self.play_sound_callback)

        # create a client for the stop listening service
        self.stop_listening_client_ = self.create_client(StopListening, self.stop_listening_service_name_.value)

        # publish to a heartbeat topic every once in awhile
        self.empty_pub_ = self.create_publisher(
            Empty,
            'speaker/heartbeat',
            10
        )
        self.spot_voice_ = self.create_publisher(
            AudioDataStamped,
            'speaker/voice',
            10
        )
        self.heartbeat_timer_ = self.create_timer(2.5, self.heartbeat_callback)

        # initialize the speaker device
        self.speaker_device_ = JackSpeakerDevice('alsa_output.pci-0000_00_1f.3.analog-stereo')
        self.speaker_device_.set_device('alsa_output.pci-0000_00_1f.3.analog-stereo')

        # for testing only
        #self.speaker_device_ = JackSpeakerDevice('alsa_output.usb-Generic_USB2.0_Device_20121120222016-00.analog-stereo')
        #self.speaker_device_.set_device('alsa_output.usb-Generic_USB2.0_Device_20121120222016-00.analog-stereo')

        # initialize the XTTS model
        self.xtts_model_ = None
        self.xtts_config_ = XttsConfig()
        self.xtts_config_.load_json(self.xtts_config_file_.value)
        self.xtts_model_ = Xtts.init_from_config(self.xtts_config_)
        self.xtts_model_.load_checkpoint(self.xtts_config_, os.path.dirname(self.xtts_config_file_.value), use_deepspeed=False)
        self.xtts_model_.cuda()
        self.get_logger().info(f"Loading XTTS Model.")

        # save speaker embeddings
        self.gpt_cond_latent, self.speaker_embedding = self.xtts_model_.get_conditioning_latents(audio_path=self.xtts_speaker_audio_file_.value, gpt_cond_len=self.xtts_model_.config.gpt_cond_len, max_ref_length=self.xtts_model_.config.max_ref_len, sound_norm_refs=self.xtts_model_.config.sound_norm_refs)


        # generate the first speech, just so the model is warmed up
        self._run_tts('Hello world! This is a test!')
        self._run_tts('I try to generate at least three sounds first')
        self._run_tts('Spot is ready to go!')
        self.seq = 0
        # self.speaker_device_.play_sound(self.xtts_output_file_.value)

    def heartbeat_callback(self) -> None:
        self.empty_pub_.publish(Empty())

    def _run_tts(self, text: str) -> None:
        """
        generates a .wav file from text
        :param text: the text whose audio we are generating
        :param path_to_wav: the path to the audio .wav file to be generated
        :return: void
        """

        # perform the inference to generate the audio
        out = self.xtts_model_.inference(
            text=text,
            language='en',
            gpt_cond_latent=self.gpt_cond_latent,
            speaker_embedding=self.speaker_embedding,
            temperature=self.xtts_model_.config.temperature,
            length_penalty=self.xtts_model_.config.length_penalty,
            repetition_penalty=self.xtts_model_.config.repetition_penalty,
            top_k=self.xtts_model_.config.top_k,
            top_p=self.xtts_model_.config.top_p
        )

        # save the audio to a file
        out['wav'] = torch.tensor(out['wav']).unsqueeze(0)
        torchaudio.save(self.xtts_output_file_.value, out['wav'], 24000)

    def _compute_wavfile_duration(self, path) -> float:
        """
        computes the duration of a wav file
        :param path: the path to the wav file
        :return: its duration in seconds as a float
        """
        wav = wave.open(path)
        frame_count = wav.getnframes()
        channel_count = wav.getnchannels()
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

        # remove the old output file
        os.remove(self.xtts_output_file_.value)

        # generate the .wav file
        self.get_logger().info("Received play sound request \"{}\"".format(request.text.lower()))
        self._run_tts(request.text.lower())

        # check that file exists
        if not os.path.exists(self.xtts_output_file_.value) or not os.path.isfile(self.xtts_output_file_.value):
            response.success = False
            return response

        # ask the microphone to stop listening for the next <duration of wavfile> seconds
        duration_s = self._compute_wavfile_duration(self.xtts_output_file_.value)

        # try to recreate the audio, if it's empty
        if duration_s == 0.0:
            self.get_logger().error(f'Generated empty audio file during transcription!')
            self._run_tts(request.text.lower())
            duration_s = self._compute_wavfile_duration(self.xtts_output_file_.value)
        # try to recreate the audio, if it's empty
        if duration_s == 0.0:
            self.get_logger().error(f'Generated empty audio file during transcription!')
            self._run_tts(request.text.lower())
            duration_s = self._compute_wavfile_duration(self.xtts_output_file_.value)

        self.get_logger().info(f"Audio Duration: {duration_s} [s]")

        stop_listening_request = StopListening.Request()
        right_now = self.get_clock().now()
        stop_listening_request.stop_listen_time = right_now.to_msg()
        stop_listening_request.start_listen_time = (right_now + Duration(seconds=duration_s + 0.5)).to_msg()
        #stop_listening_request.start_listen_time = (right_now + Duration(seconds=duration_s -0.5)).to_msg()
        self.stop_listening_client_.call_async(stop_listening_request)  # we don't even care about the result, so don't even wait for it.

        # play the sound through the speaker
        response.start_time = self.get_clock().now().to_msg()
        self.speaker_device_.play_sound(self.xtts_output_file_.value)
        response.end_time = self.get_clock().now().to_msg()

        self.spot_voice_.publish(
            AudioDataStamped(
                audio=SpeakerNode.wav_to_audio_data(self.xtts_output_file_.value),
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
