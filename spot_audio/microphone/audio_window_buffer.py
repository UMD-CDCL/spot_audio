#!/usr/bin/env python3

"""
Accumulates incoming PCM packets and hands out fixed-length analysis windows.

Split out of the node so the windowing has no ROS dependency and can be unit
tested and reused by an offline harness that reads bags directly.

Two things here are deliberate and worth not "simplifying" later:

1. Windows advance on ACCUMULATED AUDIO DURATION, never on a wall-clock timer.
   audio_classification_node.py's timers assume playback runs at 1x; replaying a
   bag at 5x under such a timer silently changes how much audio each window holds
   and how many windows a recording produces, so an evaluation run would measure
   something different from the deployed system. Counting samples makes the window
   sequence identical at any playback rate.

2. Resampling happens on whole windows, not per packet. resample_poly on each
   ~100 ms packet independently zero-pads at both edges, so the polyphase filter
   rings at every packet boundary -- ~10 discontinuities per second of audio.
   Accumulating at the source rate and decimating once per window leaves a single
   pair of edge transients per window instead.
"""

import math
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt, stft, istft
from typing import List, Optional, Tuple


def deinterleave(pcm: np.ndarray, channels: int, channel_mode: str) -> np.ndarray:
    """
    Reduces interleaved multi-channel PCM to one mono channel.

    :param pcm: interleaved int16 samples, length a multiple of `channels`
    :param channels: number of interleaved channels
    :param channel_mode: 'mix' to average all channels, or a channel index as a string
    :return: mono int16 samples, always a fresh array that owns its data

    The returned array never aliases `pcm`. Callers buffer the result for the length
    of a recording, while `pcm` is typically a view onto a ROS message payload that
    rclpy is free to reuse once the callback returns -- so a view would decay into
    whatever arrived later. The 'mix' path allocates anyway; the single-channel and
    channel-index paths would not, hence the explicit copies.
    """
    if channels == 1:
        return pcm.copy()
    usable = (len(pcm) // channels) * channels
    frames = pcm[:usable].reshape(-1, channels)
    if channel_mode == 'mix':
        # float32 intermediate so summing channels can't wrap int16
        return frames.mean(axis=1, dtype=np.float32).astype(np.int16)
    index = int(channel_mode)
    if not 0 <= index < channels:
        raise ValueError(f'channel_mode {channel_mode} out of range for {channels} channels')
    # A column of a 2-D view is strided into the source buffer; copy() detaches it.
    return frames[:, index].copy()


def resample_int16(pcm: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """
    Resamples int16 audio with polyphase filtering.

    :param pcm: mono int16 samples
    :param orig_rate: source sample rate
    :param target_rate: destination sample rate
    :return: mono int16 samples at target_rate
    """
    if orig_rate == target_rate:
        return pcm
    divisor = math.gcd(orig_rate, target_rate)
    resampled = resample_poly(pcm.astype(np.float32),
                              target_rate // divisor, orig_rate // divisor)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


class StreamingAudioWindower:
    """
    Buffers mono audio at the input rate and emits overlapping windows resampled
    to the target rate.

    Nothing is dropped: every sample stays buffered until the recording is
    finalized, so the .wav written at the end is exactly what was classified and
    the caller can prove the evaluation saw all of the audio.
    """

    def __init__(self,
                 input_rate: int,
                 target_rate: int,
                 channels: int,
                 channel_mode: str,
                 window_s: float,
                 hop_s: float,
                 min_tail_s: float):
        """
        :param input_rate: sample rate of incoming packets
        :param target_rate: sample rate the model needs
        :param channels: interleaved channel count of incoming packets
        :param channel_mode: 'mix' or a channel index as a string
        :param window_s: analysis window length in seconds
        :param hop_s: seconds between consecutive window starts (< window_s overlaps)
        :param min_tail_s: shortest leftover tail still worth classifying at finalize
        """
        self.input_rate = input_rate
        self.target_rate = target_rate
        self.channels = channels
        self.channel_mode = channel_mode
        self.window_samples = int(round(window_s * input_rate))
        self.hop_samples = int(round(hop_s * input_rate))
        self.min_tail_samples = int(round(min_tail_s * input_rate))
        if self.hop_samples <= 0 or self.window_samples <= 0:
            raise ValueError('window_s and hop_s must both be positive')

        self._chunks: List[np.ndarray] = []
        self._total_samples = 0
        # Absolute sample offset (in input-rate samples, from the first packet) where
        # the next window begins. Absolute rather than relative so overlapping hops
        # never accumulate rounding drift over a long recording.
        self._next_window_start = 0
        self._windows_emitted = 0

    @property
    def duration_s(self) -> float:
        """
        :return: seconds of audio accumulated so far
        """
        return self._total_samples / self.input_rate

    @property
    def windows_emitted(self) -> int:
        """
        :return: how many windows have been handed out since the last reset
        """
        return self._windows_emitted

    def _compact(self) -> np.ndarray:
        """
        Collapses the pending chunk list into one array.

        Appending to a list is O(1) per packet while np.concatenate on every packet
        would copy the whole recording ~10 times a second; compacting only when a
        window is actually due keeps that cost proportional to the number of windows.
        :return: the full accumulated buffer at the input rate
        """
        if len(self._chunks) > 1:
            self._chunks = [np.concatenate(self._chunks)]
        return self._chunks[0] if self._chunks else np.zeros(0, dtype=np.int16)

    def add_packet(self, pcm: np.ndarray) -> List[Tuple[float, float, np.ndarray]]:
        """
        Adds one interleaved PCM packet and returns any windows it completed.

        :param pcm: interleaved int16 samples straight off the wire
        :return: list of (start_s, end_s, mono int16 window at target_rate); usually
                 empty, occasionally more than one if a large packet spans hops
        """
        mono = deinterleave(pcm, self.channels, self.channel_mode)
        self._chunks.append(mono)
        self._total_samples += len(mono)

        windows = []
        while self._next_window_start + self.window_samples <= self._total_samples:
            buffer = self._compact()
            start = self._next_window_start
            windows.append(self._make_window(buffer, start, start + self.window_samples))
            self._next_window_start += self.hop_samples
        return windows

    def _make_window(self, buffer: np.ndarray, start: int, end: int
                     ) -> Tuple[float, float, np.ndarray]:
        """
        Slices and resamples one window out of the accumulated buffer.

        :param buffer: the compacted input-rate buffer
        :param start: first sample index of the window, in input-rate samples
        :param end: one past the last sample index, in input-rate samples
        :return: (start_s, end_s, mono int16 window at target_rate)
        """
        self._windows_emitted += 1
        return (start / self.input_rate,
                end / self.input_rate,
                resample_int16(buffer[start:end], self.input_rate, self.target_rate))

    def flush(self) -> Optional[Tuple[float, float, np.ndarray]]:
        """
        Returns the trailing partial window, if there's enough of it left to judge.

        Without this, up to hop_s of every recording -- and ALL of any recording
        shorter than window_s, of which this validation set has several -- would
        never be classified at all.
        :return: (start_s, end_s, window) or None if the tail is too short
        """
        remaining = self._total_samples - self._next_window_start
        if remaining < self.min_tail_samples:
            return None
        buffer = self._compact()
        window = self._make_window(buffer, self._next_window_start, self._total_samples)
        self._next_window_start = self._total_samples
        return window

    def full_audio(self) -> np.ndarray:
        """
        :return: everything received so far as mono int16 at target_rate -- exactly
                 the audio the windows were cut from, for writing out as a .wav
        """
        return resample_int16(self._compact(), self.input_rate, self.target_rate)

    def reset(self) -> None:
        """
        Clears all state so the next recording starts clean.
        :return: nothing
        """
        self._chunks = []
        self._total_samples = 0
        self._next_window_start = 0
        self._windows_emitted = 0


def estimate_high_pass(pcm: np.ndarray, sample_rate: int = 16000,
                       low: float = 80.0, high: float = 300.0) -> float:
    """
    Picks a high-pass corner from the clip's own spectrum instead of fixing one.

    A fixed corner is measurably wrong in both directions: 80 Hz left 92% of one
    recording's energy in place (its noise was almost entirely sub-300 Hz), while
    250 Hz destroyed a clean recording whose speech extended below it -- VAD
    segments went 2 -> 0. The right corner is a property of the clip.

    The discriminator is WHERE THE ENERGY SITS, not how stationary it is. That
    distinction was measured, not assumed: an earlier version of this function used
    a median/10th-percentile stationarity ratio on the theory that machine rumble is
    constant, and it returned the floor for every clip -- the low-band rumble turned
    out to be MORE variable (4.4-6.7) than the clean clip's speech band (2.5-2.8),
    so stationarity carries no signal here. Energy fraction separates the same two
    clips cleanly: 0.058 of total energy below 250 Hz in the clean recording against
    0.866 in the rumble-dominated one.

    Speech energy lives in 300-3400 Hz, so a clip with most of its energy below that
    is telling you the low band is noise. The corner scales with that fraction and
    is clamped to [low, high], so a clip whose voice genuinely extends low is left
    alone and one drowning in rumble gets the whole band removed.
    :param pcm: mono int16 samples
    :param sample_rate: sample rate of pcm
    :param low: corner used when the low band looks like signal
    :param high: corner used when the low band is clearly noise
    :return: corner frequency in Hz
    """
    if len(pcm) < sample_rate // 2:
        return low
    signal = pcm.astype(np.float64)
    power = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    frequencies = np.fft.rfftfreq(len(signal), 1.0 / sample_rate)
    total = power.sum()
    if total <= 0:
        return low
    low_fraction = float(power[frequencies < high].sum() / total)
    # 0.2 -> leave it alone, 0.8 -> cut the whole band, linear between.
    weight = float(np.clip((low_fraction - 0.2) / 0.6, 0.0, 1.0))
    return float(low + weight * (high - low))


def denoise(pcm: np.ndarray, sample_rate: int = 16000, high_pass_hz: float = None,
            over_subtraction: float = 1.5, floor: float = 0.08) -> np.ndarray:
    """
    Light spectral-gate denoiser for the standoff-microphone recordings.

    These are recorded by a robot several metres from the casualty in an outdoor
    scene: the spectrograms show a heavy low-frequency rumble plus a narrow tonal
    band from the machine itself, and the casualty's voice sits underneath both.
    Two stages, in order:

      1. a high-pass at 250 Hz. NOT 80 Hz, which is where this started and which was
         measurably useless here: on these recordings the machine rumble occupies
         the whole low band, and one clip measured 92% of its total energy below
         300 Hz against 7.8% in the 300-3400 Hz speech band. An 80 Hz corner left
         essentially all of that in place, which is why the first denoised run
         scored WORSE than raw. 250 Hz sits just below the male fundamental (~85-180
         Hz is lost, but its harmonics and all formants survive) and removes the
         band that is pure noise here
      2. spectral subtraction against a noise floor estimated as the per-bin 10th
         PERCENTILE over time -- a percentile rather than "the first N frames",
         because these recordings have no guaranteed silent lead-in and machine
         noise is stationary enough that a low quantile tracks it well

    `floor` keeps a fraction of the original magnitude rather than zeroing bins
    outright: aggressive gating produces musical noise, and a model asked whether
    anyone is vocalizing is exactly the wrong consumer for artefacts that sound
    like faint voices. Phase is left untouched.

    Implemented with scipy alone, deliberately -- noisereduce is not installed in
    the environment the Gemma node runs in, and adding an ML dependency there to
    do a spectral subtraction would be a poor trade.
    :param pcm: mono int16 samples
    :param sample_rate: sample rate of pcm
    :param high_pass_hz: corner frequency of the high-pass stage; None estimates it
                         per clip via estimate_high_pass(), which is the default
                         because no single fixed corner worked on both the
                         rumble-dominated and the clean recordings
    :param over_subtraction: how much of the estimated noise floor to remove; >1
                             subtracts more than the estimate, which is standard
                             practice because the estimate is biased low
    :param floor: minimum fraction of the original magnitude retained per bin
    :return: mono int16 samples, same length as the input
    """
    if len(pcm) < sample_rate // 10:
        return pcm
    signal = pcm.astype(np.float32) / 32768.0
    signal_rms_reference = signal.copy()
    if high_pass_hz is None:
        high_pass_hz = estimate_high_pass(pcm, sample_rate)

    sos = butter(4, high_pass_hz / (sample_rate / 2.0), btype='highpass', output='sos')
    signal = sosfiltfilt(sos, signal).astype(np.float32)

    frequencies, times, spectrum = stft(signal, fs=sample_rate, nperseg=512, noverlap=384)
    magnitude = np.abs(spectrum)
    noise = np.percentile(magnitude, 10, axis=1, keepdims=True)
    cleaned = np.maximum(magnitude - over_subtraction * noise, floor * magnitude)
    _, reconstructed = istft(cleaned * np.exp(1j * np.angle(spectrum)), fs=sample_rate,
                             nperseg=512, noverlap=384)

    reconstructed = reconstructed[:len(pcm)]
    if len(reconstructed) < len(pcm):
        reconstructed = np.pad(reconstructed, (0, len(pcm) - len(reconstructed)))

    # Restore the original RMS. On a rumble-dominated clip the high-pass can remove
    # ~87% of the total energy, and without this the output is not just cleaner but
    # far quieter -- which confounds "we removed noise" with "we turned it down" for
    # every downstream model that is sensitive to level, VAD most of all.
    original_rms = float(np.sqrt(np.mean(signal_rms_reference ** 2)))
    current_rms = float(np.sqrt(np.mean(reconstructed ** 2)))
    if current_rms > 1e-9 and original_rms > 1e-9:
        reconstructed = reconstructed * (original_rms / current_rms)
    return np.clip(reconstructed * 32768.0, -32768, 32767).astype(np.int16)
