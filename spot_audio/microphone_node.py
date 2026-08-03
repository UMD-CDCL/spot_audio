#!/usr/bin/env python3

"""
Publishes raw audio from the arm-mounted microphone for downstream transcription
and classification.

Receives 'AUD2' packets (see microphone/udp_microphone_device.py). The Pi
streamer and this node must be built and deployed together.

Performance note: the device layer hands this node an `array.array('B')` rather
than a numpy array. rclpy's generated setter for a uint8[] field accepts an
array.array through a fast path with no per-element validation and no copy,
which matters here since this runs on every packet (~94 times a second).
"""

import threading
import wave

from audio_common_msgs.msg import AudioData, AudioDataStamped
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from microphone.udp_microphone_device import UdpMicrophoneDevice
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Header

# Beyond this skew we stop trusting the Pi's clock and stamp on receipt instead.
MAX_CLOCK_SKEW_NS = 5_000_000_000


class MicrophoneNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name,
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        self.declare_parameter('microphone_udp_port', 21885)
        self.declare_parameter('microphone_sampling_freq_hz', 48000)  # 48000 rode, 16000 respeaker
        self.declare_parameter('main_channel', 0)
        # Seconds without audio before the stream is torn down and restarted.
        # Must be comfortably longer than the watchdog period so a single late
        # packet cannot trigger a reconnect.
        self.declare_parameter('audio_timeout_sec', 3.0)
        self.declare_parameter('conceal_packet_loss', True)

        # Cached once here rather than read via get_parameter() per packet: that
        # would take the node's parameter lock ~94 times a second from a
        # non-executor thread.
        self.main_channel = int(self.get_parameter('main_channel').value)
        self.sample_rate = int(self.get_parameter('microphone_sampling_freq_hz').value)
        self.audio_timeout_sec = float(self.get_parameter('audio_timeout_sec').value)

        # QoS left at the default reliable/depth-50 deliberately: changing it would
        # alter delivery semantics for existing subscribers.
        self.pub_raw_audio = self.create_publisher(AudioDataStamped, 'raw_audio', 50)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)

        # for saving audio to .wav
        self.save_debug_audio = False
        self.audio_buffer = bytearray()
        self.record_start_time = None

        self.microphone = UdpMicrophoneDevice(
            self.on_received_audio,
            self.get_logger(),
            self.sample_rate,
            int(self.get_parameter('microphone_udp_port').value),
            conceal_losses=bool(self.get_parameter('conceal_packet_loss').value),
        )

        self.seq = 0
        self.last_got_audio_data = None
        self.clock_skew_warned = False
        self.published_count = 0
        self.concealed_count = 0

        # Reconnects run on their own thread. stop_stream() joins two worker
        # threads, so doing it inline would block the executor for seconds and
        # stall every other callback on this node.
        self._reconnect_lock = threading.Lock()
        self._reconnecting = False

        self.watchdog_timer = self.create_timer(1.0, self.watchdog_callback)
        self.diagnostics_timer = self.create_timer(1.0, self.publish_diagnostics)

        try:
            self.microphone.start_stream()
            self.get_logger().info('Started streaming audio from the arm microphone')
        except Exception as exc:
            # Raise rather than swallow: the launch file respawns this node on
            # exit, and the Pi keeps streaming in the meantime, so failing fast
            # here is what lets the process actually recover.
            self.get_logger().fatal(f'Could not start audio stream: {exc}')
            raise

    # ------------------------------------------------------------------ stamps
    def stamp_from_capture_ns(self, capture_ns):
        """
        Convert a wire capture_ns into a ROS stamp, falling back to receive time
        if the Pi's clock is implausibly skewed from ours.
        """
        now = self.get_clock().now()
        if not capture_ns:
            return now

        capture = Time(nanoseconds=capture_ns)
        skew = now.nanoseconds - capture.nanoseconds

        if abs(skew) > MAX_CLOCK_SKEW_NS:
            if not self.clock_skew_warned:
                self.get_logger().warning(
                    f'Pi capture timestamps are {skew / 1e9:.1f}s from our clock; '
                    f'falling back to receive-time stamping. Run chrony on both '
                    f'hosts (see spot-arm-pi/README.md).')
                self.clock_skew_warned = True
            return now

        self.clock_skew_warned = False
        return capture

    # --------------------------------------------------------------- watchdog
    def watchdog_callback(self) -> None:
        if self.last_got_audio_data is None:
            return

        age = (self.get_clock().now() - self.last_got_audio_data).nanoseconds / 1e9
        if age <= self.audio_timeout_sec:
            return

        with self._reconnect_lock:
            if self._reconnecting:
                return
            self._reconnecting = True

        self.get_logger().warning(
            f'No audio for {age:.1f}s; restarting the UDP receiver')
        threading.Thread(target=self._reconnect_worker, daemon=True,
                         name='mic_reconnect').start()

    def _reconnect_worker(self) -> None:
        try:
            self.microphone.stop_stream()
        except Exception as exc:
            self.get_logger().error(f'Error stopping audio stream: {exc}')

        try:
            self.microphone.start_stream()
            # Reset the clock so the watchdog gives the fresh stream a full
            # timeout window before considering another reconnect.
            self.last_got_audio_data = self.get_clock().now()
            self.get_logger().info('UDP audio receiver restarted')
        except Exception as exc:
            self.get_logger().error(f'Error restarting audio stream: {exc}')
        finally:
            with self._reconnect_lock:
                self._reconnecting = False

    # ------------------------------------------------------------ diagnostics
    def publish_diagnostics(self) -> None:
        stats = self.microphone.get_stats()

        status = DiagnosticStatus()
        status.name = 'spot_audio: microphone udp stream'
        status.hardware_id = 'arm_microphone'

        now = self.get_clock().now()
        if self.last_got_audio_data is None:
            status.level = DiagnosticStatus.ERROR
            status.message = 'no audio received yet'
        else:
            age = (now - self.last_got_audio_data).nanoseconds / 1e9
            received = max(stats['packets_received'], 1)
            loss_pct = 100.0 * stats['packets_lost'] / received

            if age > self.audio_timeout_sec:
                status.level = DiagnosticStatus.ERROR
                status.message = f'no audio for {age:.1f}s'
            elif stats['queue_overflows'] > 0:
                status.level = DiagnosticStatus.WARN
                status.message = 'publish path falling behind (queue overflows)'
            elif loss_pct > 1.0:
                status.level = DiagnosticStatus.WARN
                status.message = f'streaming with {loss_pct:.1f}% packet loss'
            else:
                status.level = DiagnosticStatus.OK
                status.message = 'streaming'

        status.values = [
            KeyValue(key='packets_received', value=str(stats['packets_received'])),
            KeyValue(key='packets_lost', value=str(stats['packets_lost'])),
            KeyValue(key='packets_concealed', value=str(stats['packets_concealed'])),
            KeyValue(key='queue_overflows', value=str(stats['queue_overflows'])),
            KeyValue(key='messages_published', value=str(self.published_count)),
            KeyValue(key='reported_sample_rate', value=str(stats['reported_rate'])),
        ]

        msg = DiagnosticArray()
        msg.header.stamp = now.to_msg()
        msg.status = [status]
        self.diagnostics_pub.publish(msg)

    # ----------------------------------------------------------------- publish
    def on_received_audio(self, byte_array, channel, capture_ns=0,
                          concealed=False) -> None:
        """
        Publish raw audio for downstream tasks.

        :param byte_array: array.array('B') of interleaved PCM16-LE bytes
        :param channel: the channel providing the data
        :param capture_ns: CLOCK_REALTIME ns of the first sample, from the Pi
        :param concealed: True if this buffer is inserted silence, not real audio
        """
        if channel != self.main_channel:
            return

        # Concealed silence logically precedes the packet whose gap it fills, so
        # back its own duration out of that packet's capture time.
        if concealed and capture_ns and self.sample_rate > 0:
            frames = len(byte_array) // 2
            capture_ns = max(0, capture_ns - (frames * 1_000_000_000) // self.sample_rate)

        stamp = self.stamp_from_capture_ns(capture_ns)

        if self.save_debug_audio:
            if self.record_start_time is None:
                self.record_start_time = self.get_clock().now()
            self.audio_buffer.extend(byte_array)
            elapsed = (self.get_clock().now() - self.record_start_time).nanoseconds / 1e9
            if elapsed > 10.0:
                self._save_to_wav()
                self.save_debug_audio = False

        self.pub_raw_audio.publish(
            AudioDataStamped(
                header=Header(stamp=stamp.to_msg()),
                # array.array('B') hits rclpy's zero-validation fast path.
                audio=AudioData(data=byte_array),
            )
        )

        self.last_got_audio_data = self.get_clock().now()
        self.seq += 1
        self.published_count += 1
        if concealed:
            self.concealed_count += 1

    def _save_to_wav(self) -> None:
        filename = '/tmp/spot_debug_audio_10s.wav'
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(self.sample_rate)
                wf.writeframes(bytes(self.audio_buffer))
            self.get_logger().info(f'Saved 10 seconds of raw audio to {filename}')
        except Exception as exc:
            self.get_logger().error(f'Failed to save debug audio file: {exc}')
        finally:
            self.audio_buffer.clear()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = MicrophoneNode('microphone_node')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.microphone.stop_stream()
            except Exception:
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
