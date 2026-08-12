"""
UDP microphone receiver for the Spot arm.

Receives 'AUD2' packets from the Pi: a 28-byte header (magic, sequence,
capture timestamp, sample rate, channels, sample width, frame count) followed by
interleaved PCM16 little-endian audio. The Pi streamer
(spot-arm-pi/alsamic_read_audio) and this module must be built and deployed
together.

What the header provides:

  * Sequence numbers make packet loss visible and, with concealment enabled,
    recoverable in the timeline sense: a lost packet is a detectable gap rather
    than an invisible 10.7 ms discontinuity that silently shortens the stream
    and drifts it against every other sensor.
  * sample_rate on the wire catches the case where ALSA negotiated a rate other
    than the one requested, which would otherwise produce wrong-pitch audio
    with no error anywhere.
  * capture_ns lets the node stamp messages with capture time instead of
    receive time.
"""

import array
import socket
import struct
import threading
import queue
import time

# --- wire protocol -----------------------------------------------------------
# MUST match spot-arm-pi/common/spot_arm_wire.h
AUD2_MAGIC = 0x41554432          # 'AUD2'
AUD2_HEADER_FMT = '>IIQIHHI'     # magic, seq, capture_ns, rate, ch, sample_bytes, frames
AUD2_HEADER_LEN = struct.calcsize(AUD2_HEADER_FMT)
assert AUD2_HEADER_LEN == 28, "AudioHeader must stay in sync with spot_arm_wire.h"

MAX_AUDIO_FRAMES = 8192

# A gap larger than this is treated as a stream restart (or a very long outage)
# rather than something worth concealing: filling seconds of silence would just
# push a huge useless buffer downstream.
MAX_CONCEAL_PACKETS = 32


class UdpMicrophoneDevice(object):
    """Receives PCM16 mono audio over UDP and invokes on_audio(samples, channel)."""

    def __init__(self, on_audio, logger, rate=48000, port=21885,
                 conceal_losses=True):
        self.on_audio = on_audio
        self.logger = logger
        self.rate = rate
        self.port = port
        self.ip = "0.0.0.0"
        self.bitdepth = 16
        self.available_channels = 1
        self.channels = range(self.available_channels)
        # When True, detected gaps are filled with silence so the audio timeline
        # stays aligned with wall clock and with the other sensors.
        self.conceal_losses = conceal_losses

        self.sock = None
        self.net_thread = None
        self.process_thread = None
        self.running = False

        # ~2 s of audio at 512-frame packets. Shock absorber between the network
        # thread and the (slower) publish callback.
        self.audio_queue = queue.Queue(maxsize=200)

        # --- receive statistics, readable by the node for diagnostics ---
        self.stats_lock = threading.Lock()
        self.packets_received = 0
        self.packets_lost = 0
        self.packets_concealed = 0
        self.queue_overflows = 0
        self.last_packet_time = None   # time.monotonic()
        self.reported_rate = None
        self._expected_seq = None
        self._warned_rate_mismatch = False

    # ---------------------------------------------------------------- socket
    def create_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Without this, a reconnect can fail with "Address already in use" if
        # the previous socket has not been fully released yet.
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        requested = 4 * 1024 * 1024
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, requested)
            actual = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            # The request is silently clamped to net.core.rmem_max, which on a
            # stock install is ~208 KB rather than the 4 MB asked for here.
            if actual < requested:
                self.logger.warning(
                    f"SO_RCVBUF clamped to {actual} bytes (asked for {requested}). "
                    f"Raise net.core.rmem_max -- see "
                    f"spot-arm-pi/system/99-spot-arm-net.conf")
        except OSError as exc:
            self.logger.warning(f"Failed to increase OS socket buffer: {exc}")

        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(0.5)
        self.logger.info(f"Listening for AUD2 UDP audio on {self.ip}:{self.port}")

    def destroy_socket(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    # --------------------------------------------------------------- lifecycle
    def start_stream(self):
        if self.running:
            return

        if self.sock is None:
            self.create_socket()

        self.running = True
        with self.stats_lock:
            self._expected_seq = None

        self.net_thread = threading.Thread(target=self._receive_loop, daemon=True,
                                           name="mic_net")
        self.net_thread.start()

        self.process_thread = threading.Thread(target=self._process_loop, daemon=True,
                                               name="mic_process")
        self.process_thread.start()

        self.logger.info("UDP audio receiver & processing threads started.")

    def stop_stream(self):
        """
        Stop both worker threads and release the socket.

        Threads are daemons with short socket/queue timeouts, so a join that times
        out means the thread is about to exit on its own; it cannot wedge here.
        """
        if not self.running and self.sock is None:
            return

        self.running = False

        for name, thread in (("net", self.net_thread), ("process", self.process_thread)):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.5)
                if thread.is_alive():
                    self.logger.warning(
                        f"mic {name} thread did not exit within 1.5s; it will "
                        f"terminate on its own next timeout")
        self.net_thread = None
        self.process_thread = None

        self.destroy_socket()

        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self.logger.info("UDP audio receiver stopped.")

    def get_stats(self):
        """Snapshot of receive statistics for diagnostics."""
        with self.stats_lock:
            return {
                "packets_received": self.packets_received,
                "packets_lost": self.packets_lost,
                "packets_concealed": self.packets_concealed,
                "queue_overflows": self.queue_overflows,
                "last_packet_time": self.last_packet_time,
                "reported_rate": self.reported_rate,
            }

    # ------------------------------------------------------------------ threads
    def _receive_loop(self):
        """Network thread: read, validate the header, push to the queue. No decode."""
        bufsize = 4096

        while self.running:
            try:
                data, _addr = self.sock.recvfrom(bufsize)
            except socket.timeout:
                continue
            except (OSError, AttributeError) as exc:
                # AttributeError covers the race where stop_stream() nulls the
                # socket while we are between the running check and the recv.
                if self.running:
                    self.logger.error(f"UDP recv error: {exc}")
                break

            if len(data) < AUD2_HEADER_LEN:
                continue

            parsed = self._parse_packet(data)
            if parsed is None:
                continue

            try:
                self.audio_queue.put_nowait(parsed)
            except queue.Full:
                with self.stats_lock:
                    self.queue_overflows += 1
                self.logger.warning(
                    "Audio queue full; dropping packet (publish path cannot keep up)")

        self.logger.info("Network receiver loop exited.")

    def _parse_packet(self, data):
        """
        Validate one AUD2 datagram.

        Returns (pcm_bytes, capture_ns, silence_frames_to_prepend) or None.
        """
        magic, seq, capture_ns, rate, channels, sample_bytes, frames = \
            struct.unpack_from(AUD2_HEADER_FMT, data, 0)

        if magic != AUD2_MAGIC:
            return None

        if sample_bytes != 2 or channels != 1 or frames == 0 or frames > MAX_AUDIO_FRAMES:
            self.logger.warning(
                f"AUD2 header rejected: {channels}ch {sample_bytes}B "
                f"{frames} frames")
            return None

        expected_bytes = frames * channels * sample_bytes
        if len(data) < AUD2_HEADER_LEN + expected_bytes:
            self.logger.warning(
                f"AUD2 packet truncated: {len(data)} bytes for "
                f"{expected_bytes}-byte payload")
            return None

        # The Pi refuses to stream a renegotiated rate, but check anyway: a rate
        # mismatch silently transposes everything downstream.
        if rate != self.rate and not self._warned_rate_mismatch:
            self.logger.error(
                f"Microphone is streaming at {rate} Hz but this node is "
                f"configured for {self.rate} Hz. Audio will be pitch-shifted and "
                f"timing will drift. Fix microphone_sampling_freq_hz.")
            self._warned_rate_mismatch = True

        pcm = data[AUD2_HEADER_LEN:AUD2_HEADER_LEN + expected_bytes]

        # ---- sequence tracking / loss concealment ----
        conceal_frames = 0
        with self.stats_lock:
            self.packets_received += 1
            self.last_packet_time = time.monotonic()
            self.reported_rate = rate

            if self._expected_seq is not None and seq != self._expected_seq:
                gap = (seq - self._expected_seq) & 0xFFFFFFFF
                if gap < 0x80000000:
                    # Forward gap: packets were lost.
                    self.packets_lost += gap
                    if self.conceal_losses and gap <= MAX_CONCEAL_PACKETS:
                        conceal_frames = gap * frames
                        self.packets_concealed += gap
                    elif gap > MAX_CONCEAL_PACKETS:
                        self.logger.warning(
                            f"Audio gap of {gap} packets "
                            f"({gap * frames / rate:.2f}s); not concealing")
                else:
                    # Backward jump: the sender restarted and its counter reset.
                    self.logger.info("Audio sequence reset; sender restarted")

            self._expected_seq = (seq + 1) & 0xFFFFFFFF

        return (pcm, capture_ns, conceal_frames)

    def _process_loop(self):
        """Worker thread: pull from the queue and fire the callback."""
        while self.running:
            try:
                item = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            pcm, capture_ns, conceal_frames = item

            # Insert silence for the lost packets so the audio timeline stays
            # real-time aligned. Without this the stream is time-compressed on
            # loss and drifts against the other sensors.
            if conceal_frames > 0:
                silence = array.array('B', bytes(conceal_frames * 2))
                self._dispatch(silence, capture_ns, concealed=True)

            # array.array('B', bytes) is a C-level conversion, and rclpy's
            # generated setter for a uint8[] field takes an array.array straight
            # through with zero per-element validation, which matters since
            # this runs ~94 times a second.
            self._dispatch(array.array('B', pcm), capture_ns, concealed=False)

        self.logger.info("Audio processing loop exited.")

    def _dispatch(self, byte_array, capture_ns, concealed):
        try:
            for chan in self.channels:
                self.on_audio(byte_array, chan, capture_ns, concealed)
        except Exception as exc:  # noqa: BLE001 - callback is user code
            self.logger.error(f"on_audio callback error: {exc}")
