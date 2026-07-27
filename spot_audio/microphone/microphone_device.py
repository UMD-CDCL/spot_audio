from .util import ignore_stderr
import pyaudio
import numpy as np
import os


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


import threading
import pulsectl
import numpy as np
from pasimple import PaSimple, PA_STREAM_RECORD, PA_SAMPLE_S16LE


class PulseMicrophoneDevice(object):
    def __init__(self, on_audio, logger, rate=48000):
        self.on_audio = on_audio
        self.logger = logger
        self.rate = rate
        self.channels = 1
        self.device_name = None
        
        self.pa_stream = None
        self._stop_event = threading.Event()
        self._record_thread = None

    def find_device(self, query_string: str = "robot_aec_mic") -> bool:
        """
        Verifies the virtual AEC source exists on the PulseAudio server.
        """
        try:
            with pulsectl.Pulse('MicrophoneNode-Control') as pulse:
                sources = pulse.source_list()
                for source in sources:
                    if query_string in source.name:
                        self.device_name = source.name
                        self.channels = source.channel_count
                        self.logger.info(f"Successfully found PulseAudio source: {self.device_name} with {self.channels} channels.")
                        return True
                
                self.logger.error(f"Failed to find PulseAudio source matching: '{query_string}'")
                return False
        except Exception as e:
            self.logger.error(f"Error querying PulseAudio: {e}")
            return False

    def start_stream(self) -> None:
        """
        Initializes the PulseAudio stream and starts the background recording thread.
        """
        if self.device_name is None:
            self.logger.error("Cannot start stream: No device selected.")
            return

        self._stop_event.clear()
        
        # Start background reading thread
        self._record_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._record_thread.start()

    def _read_loop(self) -> None:
        """
        The background thread loop that continuously reads from the stream.
        """
        # Connect to PulseAudio
        try:
            self.pa_stream = PaSimple(
                PA_STREAM_RECORD,
                PA_SAMPLE_S16LE,
                self.channels,
                self.rate,
                app_name='MicrophoneNode',
                stream_name='Record',
                server_name=os.environ.get('PULSE_SERVER', None),
                device_name=self.device_name
            )
        except Exception as e:
            self.logger.error(f"Failed to open Pulse stream: {e}")
            return

        chunk_size = 1024
        bytes_per_sample = 2 * self.channels # 16-bit audio = 2 bytes per sample
        read_bytes = chunk_size * bytes_per_sample

        self.logger.info("PulseAudio stream started successfully.")

        try:
            while not self._stop_event.is_set():
                # Read raw bytes natively from PulseAudio
                raw_data = self.pa_stream.read(read_bytes)
                
                if raw_data:
                    # Convert to numpy array and format for the callback
                    data = np.frombuffer(raw_data, dtype=np.int16)
                    data = np.reshape(data, (chunk_size, self.channels))
                    
                    # Fire callback for main channel (assuming channel 0)
                    self.on_audio(data[:, 0], 0)
                    
        except Exception as e:
            if not self._stop_event.is_set():
                self.logger.error(f"Error reading from PulseAudio stream: {e}")
        finally:
            self._close_stream()

    def stop_stream(self) -> None:
        """
        Signals the thread to stop and cleans up the connection.
        """
        self._stop_event.set()
        if self._record_thread is not None:
            self._record_thread.join(timeout=2.0)
            self._record_thread = None

    def _close_stream(self) -> None:
        if self.pa_stream is not None:
            self.pa_stream.close()
            self.pa_stream = None

    def __del__(self):
        self.stop_stream()