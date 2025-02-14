from .util import ignore_stderr
import pyaudio
import numpy as np


# i need a logger and the name of the mic.
# is the name of the mic the only thing that differs? if so we don't even need anything like polymorphism

class MicrophoneDevice(object):
    def __init__(self, on_audio, logger, rate):
        # we allow the client to define behavior when new audio data becomes available
        self.on_audio = on_audio

        # we should use the client's logger
        self.logger = logger

        # set all device specific parameter values to default values
        self.channels = None
        self.available_channels = None
        self.device_index = None
        self.rate = rate #48000  # 16000 for ReSpeaker and 48000 for RODE NTG
        self.bitdepth = 16
        self.stream = None
        self.pyaudio = None

    def find_device(self, query_string: str) -> bool:
        """
        attempts to find the device with the given string in its name
        :return: whether the device was found or not
        """

        num_devices = self.pyaudio.get_device_count()
        self.logger.info(f"{num_devices} audio devices found.")

        # find first device matching query string
        for ii in range(num_devices):
            info = self.pyaudio.get_device_info_by_index(ii)
            name = info["name"]
            chan = info["maxInputChannels"]
            self.logger.info(f" - {ii}: {name}")
            if name.find(query_string) >= 0:
                self.available_channels = chan
                self.channels = range(self.available_channels)
                self.device_index = ii
                self.logger.info(f"\"{query_string}\" detected with {chan} channels.")
                return True

        self.logger.error("Failed to find device by name.")
        return False

    def create_pyaudio(self) -> None:
        if self.pyaudio is None:
            with ignore_stderr(enable=True):
                self.pyaudio = pyaudio.PyAudio()

    def destroy_pyaudio(self) -> None:
        if self.pyaudio is not None:
            try:
                self.pyaudio.terminate()
            except:
                pass
            finally:
                self.pyaudio = None

    def create_stream(self) -> None:
        if self.device_index is None or self.pyaudio is None:
            return
        self.stream = self.pyaudio.open(
            input=True, start=False,
            format=pyaudio.paInt16,
            channels=self.available_channels,   # why did i make this 1?
            rate=self.rate,
            frames_per_buffer=1024,
            stream_callback=self.stream_callback,
            input_device_index=self.device_index,
        )

    def destroy_stream(self) -> None:
        if self.stream is not None:
            try:
                self.stream.close()
            except:
                pass
            finally:
                self.stream = None

    def __del__(self):
        self.destroy_stream()
        self.destroy_pyaudio()

    def stream_callback(self, in_data, frame_count, time_info, status):
        # split channel
        data = np.fromstring(in_data, dtype=np.int16)
        chunk_per_channel = len(data) // self.available_channels
        data = np.reshape(data, (chunk_per_channel, self.available_channels))
        for chan in self.channels:
            chan_data = data[:, chan]
            # invoke callback
            self.on_audio(chan_data, chan)
        return None, pyaudio.paContinue

    def start_stream(self):
        if self.stream is not None:
            if self.stream.is_stopped():
                self.stream.start_stream()

    def stop_stream(self):
        if self.stream is not None:
            if self.stream.is_active():
                self.stream.stop_stream()
