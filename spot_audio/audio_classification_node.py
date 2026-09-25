#!/home/cdcl/.venv/bin/python
# NOTE ON THE SHEBANG: AST needs torch + transformers, which live in
# /home/cdcl/.venv rather than the system python3 that ROS nodes normally run
# under. ROS's own PYTHONPATH still puts rclpy and the message packages on the
# path, so this interpreter imports both stacks.
#
# WHISPER IS NOT IMPORTABLE FROM THIS INTERPRETER ON THIS MACHINE. The checkpoint
# in models/whisper is CTranslate2, and faster-whisper was deliberately installed
# into roboscout-assessment/audio_classification/venv_whisper -- with its own
# cuBLAS/cuDNN wheels -- so that the main environment every other node shares was
# left untouched. So `backends` defaults to AST only here, the node logs plainly
# when Whisper is unavailable instead of failing, and the Whisper numbers for the
# validation set come from scripts/run_whisper_baseline.sh in that venv. On a robot
# where both libraries share one environment, set backends:="['ast','whisper']".

"""
Classifies casualty audio with the AST + Whisper baseline, in the same shape as
gemma_audio_classification_node.py so the two can be compared directly.

Publishes, per analysis window, an Observation per label:
  ast_respiratory_distress  -- [P(absent), P(present)]
  ast_alertness_verbal      -- [P(normal), P(abnormal), P(absent)]
  whisper_alertness_verbal  -- [P(normal), P(abnormal), P(absent)], when Whisper runs

WHAT CHANGED FROM THE PREVIOUS VERSION OF THIS NODE, and why
  - 5 s rolling buffer on two wall-clock timers -> the same 30 s window / 15 s hop
    the Gemma node uses, advanced by SAMPLE COUNT. The old timers assumed playback
    at 1x, so replaying a bag at 5x silently changed how much audio each
    classification saw; that alone made the old node impossible to evaluate
    honestly against anything.
  - AST was run on ONE chunk of the rolling buffer and the rest discarded. It now
    sees the whole window, in the 10.24 s pieces its feature extractor accepts.
  - the AudioSet label groups were rebuilt -- the old respiratory-distress set
    included "Throat clearing" and "Sneeze" while omitting "Gasp", and verbal
    alertness had two of its three classes commented out so 'absent' could never be
    predicted at all. See microphone/ast_whisper_assessor.py.
  - noise reduction and peak-normalizing amplification are gone. The AGC blew room
    tone up to full scale during silence, which actively misleads a classifier asked
    whether anyone is vocalizing.
  - the two-layer transcript de-duplication is gone with the overlapping 1.5 s
    timer that made it necessary.
  - added: a finalize service, per-recording artifacts, a reasoning topic, and an
    ObservationDataSource carrying the full window of audio -- all matching the
    Gemma node, so one set of aggregation scripts reads both.

Live behaviour that is deliberately preserved: the stop_listening service, the
heartbeat, spot_status gating, and speech publication on 'speech'.
"""

import array
import json
import os
import queue
import random
import threading

from audio_common_msgs.msg import AudioDataStamped
from cdcl_umd_msgs.msg import Observation, ObservationDataSource, SpotStatus
from cdcl_umd_msgs.srv import StopListening
from microphone.ast_whisper_assessor import AstWhisperAssessor
from microphone.audio_window_buffer import StreamingAudioWindower
from microphone.gemma_audio_assessor import SAMPLE_RATE, pcm16_to_float32
from microphone.recording_artifacts import RecordingArtifactWriter, window_row
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

# Which ObservationModule constant each task is published under, per backend. These
# names are what the Bayesian network's emission nodes key on, so they are fixed by
# cdcl_umd_msgs/ObservationModule.msg rather than chosen here.
OBSERVATION_MODULES = {
    'ast': {'respiratory_distress': 'ast_respiratory_distress',
            'alertness_verbal': 'ast_alertness_verbal'},
    'whisper': {'alertness_verbal': 'whisper_alertness_verbal'},
}

_SHUTDOWN = object()


class AudioClassificationNode(Node):
    def __init__(self, node_name: str = 'audio_classification_node'):
        super().__init__(node_name)

        self.audio_callback_group = MutuallyExclusiveCallbackGroup()
        self.service_callback_group = MutuallyExclusiveCallbackGroup()

        # ---------------------------------------------------------------- models
        self.declare_parameter('path_to_classifier',
                               '/home/cdcl/cdcl_ws/models/alertness_verbal_classifiers/')
        self.declare_parameter('path_to_whisper', '/home/cdcl/cdcl_ws/models/whisper/')
        # See the shebang comment: Whisper is not importable from this interpreter on
        # this machine, so AST alone is the default and asking for Whisper degrades
        # with a warning rather than refusing to start.
        self.declare_parameter('backends', ['ast'])
        self.declare_parameter('device', 'cuda')
        # Whisper's Silero VAD. The previous node ran with it ON, which is the honest
        # default to keep -- but it rejects most of this validation set outright, and
        # a rejected window is a silent prediction of 'absent'.
        self.declare_parameter('whisper_vad', True)
        self.declare_parameter('publish_raw_audio', True)

        # ---------------------------------------------------------------- audio
        self.declare_parameter('audio_topic', 'raw_audio')
        self.declare_parameter('input_sample_rate', 48000)
        self.declare_parameter('input_channels', 2)
        self.declare_parameter('channel_mode', 'mix')
        self.declare_parameter('window_s', 30.0)
        self.declare_parameter('hop_s', 15.0)
        self.declare_parameter('min_tail_s', 1.0)

        # ------------------------------------------------------------ behaviour
        self.declare_parameter('require_assessing', True)
        self.declare_parameter('platform_name', os.environ.get('SPOT_NAME', 'unknown'))

        # ------------------------------------------------------------- artifacts
        self.declare_parameter('save_artifacts', False)
        self.declare_parameter('output_dir', '')
        self.declare_parameter('recording_name', '')
        self.declare_parameter('labels_json', '')

        self.platform_name = self.get_parameter('platform_name').value
        requested = [str(b) for b in self.get_parameter('backends').value]

        self.windower = StreamingAudioWindower(
            input_rate=int(self.get_parameter('input_sample_rate').value),
            target_rate=SAMPLE_RATE,
            channels=int(self.get_parameter('input_channels').value),
            channel_mode=str(self.get_parameter('channel_mode').value),
            window_s=float(self.get_parameter('window_s').value),
            hop_s=float(self.get_parameter('hop_s').value),
            min_tail_s=float(self.get_parameter('min_tail_s').value),
        )

        self.assessor = AstWhisperAssessor(
            ast_path=str(self.get_parameter('path_to_classifier').value),
            whisper_path=str(self.get_parameter('path_to_whisper').value),
            backends=requested,
            device=str(self.get_parameter('device').value),
            whisper_vad=bool(self.get_parameter('whisper_vad').value),
            logger=self.get_logger(),
        )
        try:
            self.assessor.load()
        except ImportError as exc:
            # Losing Whisper should cost the Whisper numbers, not the whole node.
            if 'whisper' not in requested:
                raise
            self.get_logger().warning(
                f'Whisper backend unavailable in this interpreter ({exc}); continuing '
                f'with AST only. Whisper lives in venv_whisper -- see '
                f'scripts/run_whisper_baseline.sh.')
            self.assessor.backends = tuple(b for b in requested if b != 'whisper')
            self.assessor.load()
        self.tasks = self.assessor.supported_tasks()
        self.active_backends = tuple(self.assessor.backends)

        # Unbounded on purpose: dropping audio would make an evaluation silently
        # measure less than the whole recording. Finalize waits for the backlog.
        self.work_queue = queue.Queue()
        self.max_backlog = 0
        self.rows = []
        self.notes = []
        self.recording_started_at = None
        self.rows_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True,
                                       name='ast_whisper_inference')
        self.worker.start()

        self.sub_audio_data = self.create_subscription(
            AudioDataStamped, str(self.get_parameter('audio_topic').value),
            self.audio_data_callback, 50, callback_group=self.audio_callback_group)
        self.sub_spot_status = self.create_subscription(
            SpotStatus, 'spot_status', self.spot_status_callback, qos_profile_sensor_data)
        self.assessing = not bool(self.get_parameter('require_assessing').value)

        self.pub_observation = self.create_publisher(Observation, 'observation_no_id', 10)
        self.pub_observation_data_source = self.create_publisher(
            ObservationDataSource, 'observation_data_sources', 10)
        self.pub_speech = self.create_publisher(ObservationDataSource, 'speech', 10)
        self.pub_reasoning = self.create_publisher(
            String, 'audio_classification/reasoning', 10)
        self.pub_heartbeat = self.create_publisher(
            Empty, 'audio_classification/heartbeat', 10)
        self.heartbeat_timer = self.create_timer(2.5, self.heartbeat_callback)

        # Preserved from the previous node: the speaker calls this before it talks so
        # the robot never transcribes its own voice.
        self.stop_listening_service = self.create_service(
            StopListening, 'stop_listening', self.stop_listening_callback)
        self.stop_listening_start_time = None
        self.stop_listening_stop_time = None

        self.finalize_service = self.create_service(
            Trigger, '~/finalize_recording', self.finalize_callback,
            callback_group=self.service_callback_group)

        self.get_logger().info(
            f"Ready. backends={list(self.active_backends)} "
            f"tasks={[t.name for t in self.tasks]} "
            f"topic={self.get_parameter('audio_topic').value} "
            f"window={self.get_parameter('window_s').value}s "
            f"hop={self.get_parameter('hop_s').value}s "
            f"require_assessing={self.get_parameter('require_assessing').value}")

    # --------------------------------------------------------------- callbacks
    def heartbeat_callback(self) -> None:
        self.pub_heartbeat.publish(Empty())

    def spot_status_callback(self, msg: SpotStatus) -> None:
        if not bool(self.get_parameter('require_assessing').value):
            return
        self.assessing = msg.state == SpotStatus.ASSESSING

    def stop_listening_callback(self, request, response):
        """
        Records the window during which the robot is speaking, so its own voice is
        muted rather than classified as the casualty's.
        :return: the populated response
        """
        self.stop_listening_start_time = Time.from_msg(request.stop_listen_time)
        self.stop_listening_stop_time = Time.from_msg(request.start_listen_time)
        response.success = True
        self.get_logger().info(
            f'Muting audio between {self.stop_listening_start_time} and '
            f'{self.stop_listening_stop_time}')
        return response

    def audio_data_callback(self, msg: AudioDataStamped) -> None:
        """
        Buffers one packet and queues any window it completed.
        :return: nothing
        """
        if not self.assessing:
            return
        pcm = np.frombuffer(memoryview(msg.audio.data), dtype=np.int16)
        if (self.stop_listening_start_time is not None
                and self.stop_listening_stop_time is not None
                and self.stop_listening_start_time <= self.get_clock().now()
                <= self.stop_listening_stop_time):
            pcm = np.zeros_like(pcm)
        with self.buffer_lock:
            if self.recording_started_at is None:
                self.recording_started_at = self.get_clock().now().nanoseconds / 1e9
            windows = self.windower.add_packet(pcm)
        for window in windows:
            self.work_queue.put(window)
        self.max_backlog = max(self.max_backlog, self.work_queue.qsize())

    def _worker_loop(self) -> None:
        while True:
            item = self.work_queue.get()
            try:
                if item is _SHUTDOWN:
                    return
                self._process_window(*item)
            except Exception as exc:
                self.get_logger().error(f'Window scoring failed: {exc}')
            finally:
                self.work_queue.task_done()

    def _process_window(self, start_s: float, end_s: float, pcm: np.ndarray) -> None:
        """
        Scores one window, publishes its observations, and records a result row.
        :return: nothing
        """
        audio = pcm16_to_float32(pcm)
        assessment = self.assessor.assess_window(audio)

        data_source = ObservationDataSource(
            data_source_id=random.randint(-2**31, 2**31 - 1),
            # array.array('B') hits rclpy's zero-validation fast path for uint8[].
            raw_audio=(array.array('B', pcm.tobytes())
                       if bool(self.get_parameter('publish_raw_audio').value)
                       else array.array('B')),
            platform_name=self.platform_name,
            audio_transcript=assessment.note or '',
        )
        stamp = self.get_clock().now().to_msg()
        for task in self.tasks:
            for backend in self.active_backends:
                module = OBSERVATION_MODULES.get(backend, {}).get(task.name)
                if module is None:
                    continue
                self.pub_observation.publish(Observation(
                    stamp=stamp,
                    platform_name=self.platform_name,
                    data_source_id=data_source.data_source_id,
                    observation_module=module,
                    observation=assessment.probabilities[task.name].tolist(),
                    confidence=float(np.max(assessment.probabilities[task.name])),
                ))
        self.pub_observation_data_source.publish(data_source)

        # Whisper's transcript is the only speech this node produces; preserved for
        # the conversation modules that consumed it from the previous version.
        if assessment.note:
            self.pub_speech.publish(ObservationDataSource(
                data_source_id=data_source.data_source_id,
                platform_name=self.platform_name,
                audio_transcript=assessment.note))

        self.pub_reasoning.publish(String(data=json.dumps({
            'stamp': stamp.sec + stamp.nanosec * 1e-9,
            'recording_name': str(self.get_parameter('recording_name').value),
            'data_source_id': data_source.data_source_id,
            'window_index': len(self.rows),
            'start_s': round(start_s, 3),
            'end_s': round(end_s, 3),
            'backends': list(self.active_backends),
            'transcript': assessment.note or '',
            'probabilities': {task.name: {c: round(float(p), 6) for c, p in
                                          zip(task.classes,
                                              assessment.probabilities[task.name])}
                              for task in self.tasks},
            'latency_s': {k: round(v, 4) for k, v in assessment.latency_s.items()},
        }, sort_keys=True)))

        row = window_row(len(self.rows), start_s, end_s, audio, assessment, self.tasks,
                         note_field='transcript')
        with self.rows_lock:
            self.rows.append(row)
            if assessment.note:
                self.notes.append((start_s, end_s, assessment.note))

        self.get_logger().info(
            f'[{start_s:6.1f}-{end_s:6.1f}s] ' + '  '.join(
                f'{task.name}={row[f"{task.name}_pred"]}' for task in self.tasks)
            + f'  {row["latency_s"]:.2f}s')

    def finalize_callback(self, request, response):
        """
        Closes out the current recording: flushes the tail window, drains the
        backlog, writes artifacts, and resets. Synchronous, so an evaluation harness
        knows a recording is complete before the next one starts playing.
        :return: the populated response
        """
        with self.buffer_lock:
            tail = self.windower.flush()
            duration_s = self.windower.duration_s
            audio = self.windower.full_audio() if duration_s > 0 else np.zeros(0, np.int16)
        if tail is not None:
            self.work_queue.put(tail)
        self.work_queue.join()

        with self.rows_lock:
            rows = list(self.rows)
            notes = list(self.notes)
        if not rows:
            response.success = False
            response.message = (f'No windows classified ({duration_s:.1f}s of audio '
                                f'received); nothing written.')
            self.get_logger().warning(response.message)
            self._reset()
            return response

        message = f'{len(rows)} window(s) over {duration_s:.1f}s'
        if bool(self.get_parameter('save_artifacts').value):
            try:
                writer = RecordingArtifactWriter(
                    tasks=self.tasks,
                    output_dir=str(self.get_parameter('output_dir').value),
                    recording_name=str(self.get_parameter('recording_name').value),
                    labels_json=str(self.get_parameter('labels_json').value),
                    algorithm='+'.join(self.active_backends),
                    metadata={'window_s': self.get_parameter('window_s').value,
                              'hop_s': self.get_parameter('hop_s').value,
                              'channel_mode': self.get_parameter('channel_mode').value,
                              'whisper_vad': self.get_parameter('whisper_vad').value,
                              'max_inference_backlog': self.max_backlog})
                notes_text = '\n\n'.join(
                    f'[{start:.0f}-{end:.0f}s]\n{text}' for start, end, text in notes)
                directory = writer.write(rows, audio, duration_s, notes_text)
                message += f'; artifacts in {directory}'
            except Exception as exc:
                self.get_logger().error(f'Failed writing artifacts: {exc}')
                response.success = False
                response.message = f'{message}; ARTIFACT WRITE FAILED: {exc}'
                self._reset()
                return response

        response.success = True
        response.message = message
        self.get_logger().info(f'Finalized: {message}')
        self._reset()
        return response

    def _reset(self) -> None:
        with self.buffer_lock:
            self.windower.reset()
            self.recording_started_at = None
        with self.rows_lock:
            self.rows = []
            self.notes = []
        self.max_backlog = 0

    def shutdown(self) -> None:
        self.work_queue.put(_SHUTDOWN)
        self.worker.join(timeout=5.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = AudioClassificationNode()
        rclpy.spin(node, executor=MultiThreadedExecutor())
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
