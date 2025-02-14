from contextlib import contextmanager
import numpy as np
import numpy.typing as npt
import os
from scipy.io import wavfile
import sys


@contextmanager
def ignore_stderr(enable=True):
    """
    Suppress error messages from ALSA
    https://stackoverflow.com/questions/7088672/pyaudio-working-but-spits-out-error-messages-each-time
    https://stackoverflow.com/questions/36956083/how-can-the-terminal-output-of-executables-run-by-python-functions-be-silenced-i
    :param enable:
    """
    if enable:
        devnull = None
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            stderr = os.dup(2)
            sys.stderr.flush()
            os.dup2(devnull, 2)
            try:
                yield
            finally:
                os.dup2(stderr, 2)
                os.close(stderr)
        finally:
            if devnull is not None:
                os.close(devnull)
    else:
        yield


def save_audio_to_wav(raw_audio_data: npt.NDArray, rate: int, file_name: str):
    wavfile.write(file_name, rate, raw_audio_data)


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
