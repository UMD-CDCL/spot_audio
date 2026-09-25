#!/home/cdcl/.venv/bin/python
# NOTE ON THE SHEBANG: this node needs torch + transformers >= 5.12 (for the
# gemma4_unified architecture), which live in /home/cdcl/.venv, not in the system
# python3 that ROS nodes normally run under. /home/cdcl/.venv is not a
# --system-site-packages venv, but ROS's own PYTHONPATH still puts rclpy and the
# message packages on the path, so this interpreter imports both stacks; that
# combination is verified to deserialize AudioDataStamped correctly despite the
# venv's numpy 2.x. Point this at a different interpreter and the model import
# fails immediately rather than subtly.

"""
Classifies casualty audio with a single unified Gemma 4 model, replacing the
AST + Whisper pipeline in audio_classification_node.py.

Subscribes to AudioDataStamped and publishes, per analysis window, an Observation for
verbal alertness:
  gemma_audio_alertness_verbal     -- [P(normal), P(abnormal), P(absent)]

Respiratory distress is NOT assessed here. AST does it in
audio_classification_node.py, and asking Gemma for it in the same notes was measured
to damage verbal alertness badly enough that the breathing questions were removed --
see NOTE_PROMPT in microphone/gemma_audio_assessor.py.

Probabilities come from logit scoring, not parsed text -- see
microphone/gemma_audio_assessor.py.

WHAT THIS DROPS FROM THE OLD NODE, and why none of it is needed here:
  - the 5 s noise buffer and noisereduce pass: scaffolding for Whisper's
    sensitivity to background noise; the audio encoder handles noisy input.
  - peak-normalizing amplification: an AGC that blows room tone up to full scale
    during silence, which actively misleads a model asked whether anyone is
    vocalizing.
  - the two-layer transcript de-duplication (timestamp window + trailing-word
    comparison): only existed because a 1.5 s timer re-decoded an overlapping 5 s
    window. Classification windows are scored independently, so overlap is not a
    correctness problem, just a cost.
  - chopping the window into chunks and classifying only the last one.

WHAT IT ADDS:
  - correct handling of multi-channel input (the validation set is 48 kHz STEREO;
    the old node's .view(np.int16) treats interleaved stereo as mono).
  - a finalize service that writes this recording's artifacts and resets state, so
    an evaluation harness gets a synchronous guarantee that a bag is complete
    before the next one starts playing.
  - a worker thread for inference, so a slow forward pass applies backpressure to
    a queue instead of dropping incoming audio.
  - a per-window trace under <output_dir>/<recording>/trace/windows/ when
    save_artifacts is on: wNN.wav is exactly the audio that window was judged on, and
    wNN.json holds the prompt, the note, the robot's own words and the probabilities.
    Every published number can therefore be listened to and argued with afterwards.

PUBLISHED TOPICS
  observation_no_id              Observation, one per task per window
  observation_data_sources       ObservationDataSource carrying the FULL window of
                                 16 kHz mono PCM those Observations were derived
                                 from, under the same data_source_id
  audio_classification/reasoning String of JSON -- the model's working, not just its
                                 answer. One message per window:
                                   {stamp, recording_name, data_source_id,
                                    window_index, start_s, end_s, note,
                                    robot_speech[], grounded_in_note,
                                    probabilities{task{class: p}}, latency_s{}}
                                 plus one per recording with kind="recording_verdict"
                                 carrying the concatenated notes and the final
                                 probabilities. data_source_id ties a trace to the
                                 exact audio and Observations it produced, so any
                                 number can be traced back to the note behind it.

SUBSCRIBED TOPICS
  raw_audio                      AudioDataStamped (parameter: audio_topic)
  spot_status                    gates processing on ASSESSING
  speaker/spoken_text            String, what the robot itself just said (parameter:
                                 robot_speech_topic; published by speaker_node). On
                                 the real platform the robot's own words are KNOWN,
                                 so they are handed to the model verbatim rather than
                                 leaving it to recognize a synthetic voice --
                                 mis-attributing the robot's prompts to the casualty
                                 is the largest measured error source in this task.
"""

import array
import csv
import json
import os
import queue
import random
import threading
import wave

from audio_common_msgs.msg import AudioDataStamped
from cdcl_umd_msgs.msg import Observation, ObservationDataSource, SpotStatus
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from microphone.audio_window_buffer import StreamingAudioWindower
from microphone.gemma_audio_assessor import (
    GEMMA_TASKS,
    GemmaAudioAssessor,
    SAMPLE_RATE,
    pcm16_to_float32,
    pool_probabilities,
)
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.signal import spectrogram
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

# Sentinel pushed onto the work queue to tell the worker thread to exit.
_SHUTDOWN = object()


class GemmaAudioClassificationNode(Node):
    def __init__(self, node_name: str = 'gemma_audio_classification_node'):
        # Every parameter is declared explicitly below, so this deliberately does NOT
        # use the old node's allow_undeclared/automatically_declare_from_overrides
        # pair: with those on, a launch file that sets a parameter pre-declares it and
        # the explicit declare_parameter call below then raises
        # ParameterAlreadyDeclaredException.
        super().__init__(node_name)

        # Audio ingest runs on its own callback group so a long forward pass can
        # never stall packet reception (the worker thread below is what actually
        # decouples them; this keeps the service callable meanwhile).
        self.audio_callback_group = MutuallyExclusiveCallbackGroup()
        self.service_callback_group = MutuallyExclusiveCallbackGroup()

        # ---------------------------------------------------------------- model
        self.declare_parameter('model_id', 'google/gemma-4-12B-it')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('dtype', 'bfloat16')
        # Two-stage assessment. With this on, every window also produces structured
        # observation notes, and finalize reaches ONE verdict per label from the
        # whole recording's notes (see GemmaAudioAssessor.assess_recording). That is
        # the only path that can apply a rubric phrased in terms of attempts --
        # 'responsive after at most 2 attempts', 'no vocalization after 2 speech
        # prompts' -- because those count events across the encounter, not within a
        # 30 s window. Costs roughly 2.5 s per window on top of the ~0.3 s scoring.
        self.declare_parameter('take_notes', True)
        # Score each question conditioned on the note this window already produced,
        # rather than on the audio alone. Costs nothing extra when take_notes is on
        # (the note is written either way) and is the intervention that makes the
        # model commit in writing to who said what before it is asked to judge.
        self.declare_parameter('ground_in_note', True)
        # Topic carrying what the robot itself said, published by speaker_node when it
        # speaks. Verified text beats asking the model to recognize a synthetic voice.
        self.declare_parameter('robot_speech_topic', 'speaker/spoken_text')
        # Attaching the window's audio to every ObservationDataSource means turning
        # 30 s of 16 kHz PCM into a ~960 KB uint8 field per window. The deployed
        # stack wants it (downstream modules re-read the audio); an evaluation sweep
        # only pays for it, and heard.wav already preserves the audio on disk.
        self.declare_parameter('publish_raw_audio', True)

        # ---------------------------------------------------------------- audio
        self.declare_parameter('audio_topic', 'raw_audio')
        self.declare_parameter('input_sample_rate', 48000)
        self.declare_parameter('input_channels', 2)
        # 'mix' averages channels; a numeric string picks one. The validation set's
        # two channels correlate at ~0.79, so they are genuinely different signals
        # and which one is used is a real experimental choice, not a formality.
        self.declare_parameter('channel_mode', 'mix')
        self.declare_parameter('window_s', 30.0)
        self.declare_parameter('hop_s', 15.0)
        # Shortest trailing remainder still worth classifying. 1.0 s rather than
        # something more conservative because a clip shorter than this produces NO
        # windows at all and drops out of the evaluation silently -- and the
        # validation set has 16 clips under 5 s (shortest 2.0 s), which are unlikely
        # to be a representative 8% to lose.
        self.declare_parameter('min_tail_s', 1.0)

        # ------------------------------------------------------------ behaviour
        # The deployed robot only listens while assessing; the validation bags have
        # no spot_status topic at all, so evaluation runs turn this off.
        self.declare_parameter('require_assessing', True)
        self.declare_parameter('platform_name', os.environ.get('SPOT_NAME', 'unknown'))

        # ------------------------------------------------------------- artifacts
        self.declare_parameter('save_artifacts', False)
        self.declare_parameter('output_dir', '')
        self.declare_parameter('recording_name', '')
        # Optional COCO-style labels.json; when set, ground truth for the current
        # recording is looked up and drawn onto this recording's figures.
        self.declare_parameter('labels_json', '')

        self.platform_name = self.get_parameter('platform_name').value
        # Verbal alertness only; respiratory distress is AST's job in
        # audio_classification_node.py.
        self.tasks = GEMMA_TASKS

        self.windower = StreamingAudioWindower(
            input_rate=int(self.get_parameter('input_sample_rate').value),
            target_rate=SAMPLE_RATE,
            channels=int(self.get_parameter('input_channels').value),
            channel_mode=str(self.get_parameter('channel_mode').value),
            window_s=float(self.get_parameter('window_s').value),
            hop_s=float(self.get_parameter('hop_s').value),
            min_tail_s=float(self.get_parameter('min_tail_s').value),
        )

        self.assessor = GemmaAudioAssessor(
            model_id=str(self.get_parameter('model_id').value),
            tasks=self.tasks,
            device=str(self.get_parameter('device').value),
            dtype=str(self.get_parameter('dtype').value),
            logger=self.get_logger(),
        )
        self.assessor.load()

        # Unbounded on purpose: dropping audio would make an evaluation silently
        # measure less than the whole recording. If inference can't keep up with
        # playback the backlog grows and finalize waits for it -- slow, but honest.
        # max_backlog is recorded per recording so a run that needed that slack is
        # visible in the output rather than invisible.
        self.work_queue = queue.Queue()
        self.max_backlog = 0
        self.rows = []
        # Per-window audio + prompt + note, kept only while save_artifacts is on
        self.window_traces = []
        # (start_s, end_s, note) per window, in time order -- kept unflattened because
        # assess_recording() reads it as prose, while self.rows holds the CSV-safe form.
        self.notes = []
        # Wall-clock time of this recording's first audio packet; window offsets are
        # relative to it, and robot utterances are absolute.
        self.recording_started_at = None
        self.rows_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True,
                                       name='gemma_inference')
        self.worker.start()

        self.sub_audio_data = self.create_subscription(
            AudioDataStamped,
            str(self.get_parameter('audio_topic').value),
            self.audio_data_callback,
            50,
            callback_group=self.audio_callback_group,
        )
        self.sub_spot_status = self.create_subscription(
            SpotStatus, 'spot_status', self.spot_status_callback, qos_profile_sensor_data)
        # (stamp_seconds, text) for everything the robot has said, trimmed to what can
        # still overlap a live window.
        self.robot_speech = []
        self.robot_speech_lock = threading.Lock()
        self.sub_robot_speech = self.create_subscription(
            String, str(self.get_parameter('robot_speech_topic').value),
            self.robot_speech_callback, 10)
        self.assessing = not bool(self.get_parameter('require_assessing').value)

        self.pub_observation = self.create_publisher(Observation, 'observation_no_id', 10)
        self.pub_observation_data_source = self.create_publisher(
            ObservationDataSource, 'observation_data_sources', 10)
        # Everything the model wrote on its way to a number. Published as JSON in a
        # String so it can be recorded, grepped and replayed without a new message
        # package: a probability with no record of the reasoning behind it cannot be
        # audited after the fact, and these notes are exactly where mis-attribution
        # becomes visible.
        self.pub_reasoning = self.create_publisher(
            String, 'audio_classification/reasoning', 10)
        self.pub_heartbeat = self.create_publisher(Empty, 'audio_classification/heartbeat', 10)
        self.heartbeat_timer = self.create_timer(2.5, self.heartbeat_callback)

        # Private namespace ('~/finalize_recording' -> '/<node>/finalize_recording')
        # so this can never collide with another node's reset service under the
        # same namespace.
        self.finalize_service = self.create_service(
            Trigger, '~/finalize_recording', self.finalize_callback,
            callback_group=self.service_callback_group)

        self.get_logger().info(
            f"Ready. model={self.assessor.model_id} "
            f"topic={self.get_parameter('audio_topic').value} "
            f"window={self.get_parameter('window_s').value}s "
            f"hop={self.get_parameter('hop_s').value}s "
            f"channels={self.get_parameter('input_channels').value}"
            f"/{self.get_parameter('channel_mode').value} "
            f"require_assessing={self.get_parameter('require_assessing').value}")

    # --------------------------------------------------------------- callbacks
    def heartbeat_callback(self) -> None:
        """
        Tells the spot status publisher this node is alive.
        :return: nothing
        """
        self.pub_heartbeat.publish(Empty())

    def spot_status_callback(self, msg: SpotStatus) -> None:
        """
        Gates processing on the robot actually assessing a casualty.
        :param msg: the spot status message
        :return: nothing
        """
        if not bool(self.get_parameter('require_assessing').value):
            return
        self.assessing = msg.state == SpotStatus.ASSESSING

    def robot_speech_callback(self, msg: String) -> None:
        """
        Records what the robot just said, stamped on arrival.

        Arrival time rather than the PlaySound start_time because this topic carries
        only the text; the stamp is good to well within a 30 s window, which is all
        the resolution the lookup below needs.
        :param msg: the spoken text
        :return: nothing
        """
        now = self.get_clock().now().nanoseconds / 1e9
        with self.robot_speech_lock:
            self.robot_speech.append((now, msg.data))
            # Nothing older than one window can still overlap a window being scored.
            horizon = now - 4.0 * float(self.get_parameter('window_s').value)
            self.robot_speech = [(t, text) for t, text in self.robot_speech if t >= horizon]

    def robot_speech_in_window(self, start_s: float, end_s: float):
        """
        :param start_s: window start, seconds since this recording's first packet
        :param end_s: window end, same clock
        :return: verbatim robot utterances overlapping this window, oldest first
        """
        if self.recording_started_at is None:
            return []
        with self.robot_speech_lock:
            entries = list(self.robot_speech)
        # Window times are relative to the first audio packet; utterance times are
        # absolute, so shift one onto the other rather than comparing across clocks.
        window_start = self.recording_started_at + start_s
        window_end = self.recording_started_at + end_s
        return [text for stamp, text in entries if window_start <= stamp <= window_end]

    def audio_data_callback(self, msg: AudioDataStamped) -> None:
        """
        Buffers one packet and queues any analysis window it completed.

        Only the cheap buffering happens here; scoring runs on the worker thread so
        this callback returns in microseconds no matter how slow the model is.
        :param msg: raw audio packet
        :return: nothing
        """
        if not self.assessing:
            return
        # np.frombuffer over the array.array avoids copying the packet twice.
        pcm = np.frombuffer(memoryview(msg.audio.data), dtype=np.int16)
        with self.buffer_lock:
            if self.recording_started_at is None:
                self.recording_started_at = self.get_clock().now().nanoseconds / 1e9
            windows = self.windower.add_packet(pcm)
        for window in windows:
            self.work_queue.put(window)
        self.max_backlog = max(self.max_backlog, self.work_queue.qsize())

    def _worker_loop(self) -> None:
        """
        Consumes analysis windows and scores them until shutdown.
        :return: nothing
        """
        while True:
            item = self.work_queue.get()
            try:
                if item is _SHUTDOWN:
                    return
                self._process_window(*item)
            except Exception as exc:  # keep one bad window from killing the run
                self.get_logger().error(f'Window scoring failed: {exc}')
            finally:
                self.work_queue.task_done()

    def _process_window(self, start_s: float, end_s: float, pcm: np.ndarray) -> None:
        """
        Scores one window, publishes its observations, and records a result row.

        :param start_s: window start, seconds since the first packet of this recording
        :param end_s: window end, seconds since the first packet of this recording
        :param pcm: mono int16 window at SAMPLE_RATE
        :return: nothing
        """
        audio = pcm16_to_float32(pcm)
        take_notes = bool(self.get_parameter('take_notes').value)
        robot_speech = self.robot_speech_in_window(start_s, end_s)
        assessment = self.assessor.assess_window(
            audio, take_notes=take_notes,
            ground_in_note=take_notes and bool(self.get_parameter('ground_in_note').value),
            robot_speech=robot_speech)

        data_source = ObservationDataSource(
            data_source_id=random.randint(-2**31, 2**31 - 1),
            # array.array('B') hits rclpy's zero-validation fast path for a uint8[]
            # field; a numpy array or list is validated element by element, which on
            # ~960 KB of audio is the single most expensive thing in this callback.
            raw_audio=(array.array('B', pcm.tobytes())
                       if bool(self.get_parameter('publish_raw_audio').value)
                       else array.array('B')),
            platform_name=self.platform_name,
            audio_transcript=assessment.note or '',
        )
        stamp = self.get_clock().now().to_msg()
        for task in self.tasks:
            self.pub_observation.publish(Observation(
                stamp=stamp,
                platform_name=self.platform_name,
                data_source_id=data_source.data_source_id,
                observation_module=task.observation_module,
                observation=assessment.probabilities[task.name].tolist(),
                confidence=assessment.confidence(task),
            ))
        self.pub_observation_data_source.publish(data_source)

        # The reasoning trace, tied to the same data_source_id as the Observations it
        # produced, so a probability can always be traced back to the note behind it.
        self.pub_reasoning.publish(String(data=json.dumps({
            'stamp': stamp.sec + stamp.nanosec * 1e-9,
            'recording_name': str(self.get_parameter('recording_name').value),
            'data_source_id': data_source.data_source_id,
            'window_index': len(self.rows),
            'start_s': round(start_s, 3),
            'end_s': round(end_s, 3),
            'note': assessment.note or '',
            'robot_speech': robot_speech,
            'grounded_in_note': bool(assessment.note) and bool(
                self.get_parameter('ground_in_note').value),
            'probabilities': {task.name: {c: round(float(p), 6) for c, p in
                                          zip(task.classes,
                                              assessment.probabilities[task.name])}
                              for task in self.tasks},
            'latency_s': {k: round(v, 4) for k, v in assessment.latency_s.items()},
        }, sort_keys=True)))

        row = {
            'window_index': len(self.rows),
            'start_s': round(start_s, 3),
            'end_s': round(end_s, 3),
            'duration_s': round(end_s - start_s, 3),
            'rms': round(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))), 6),
            'peak': round(float(np.max(np.abs(audio))), 6),
            'latency_s': round(sum(assessment.latency_s.values()), 4),
            # Newlines flattened so windows.csv stays one physical line per window and
            # survives grep/head; notes.txt keeps the readable multi-line form.
            'note': ' | '.join((assessment.note or '').splitlines()),
        }
        for task in self.tasks:
            probs = assessment.probabilities[task.name]
            for class_name, p in zip(task.classes, probs):
                row[f'{task.name}_p_{class_name}'] = round(float(p), 6)
            row[f'{task.name}_pred'] = assessment.label(task)
        with self.rows_lock:
            self.rows.append(row)
            if assessment.note:
                self.notes.append((start_s, end_s, assessment.note))
            if bool(self.get_parameter('save_artifacts').value):
                # Held only while artifacts are on: 30 s of 16 kHz PCM is ~960 KB per
                # window, worth keeping for a trace someone can listen to and argue
                # with, and not worth keeping otherwise.
                self.window_traces.append({
                    'window_index': row['window_index'],
                    'start_s': row['start_s'], 'end_s': row['end_s'],
                    'data_source_id': data_source.data_source_id,
                    'note': assessment.note or '',
                    'note_prompt': self.assessor.note_prompt,
                    'robot_speech': robot_speech,
                    'probabilities': {
                        task.name: {c: round(float(p), 6) for c, p in
                                    zip(task.classes, assessment.probabilities[task.name])}
                        for task in self.tasks},
                    'latency_s': {k: round(v, 4) for k, v in assessment.latency_s.items()},
                    'audio': pcm.copy(),
                })

        self.get_logger().info(
            f"[{start_s:6.1f}-{end_s:6.1f}s] " + '  '.join(
                f"{task.name}={assessment.label(task)}({assessment.confidence(task):.2f})"
                for task in self.tasks) +
            f"  {row['latency_s']:.2f}s"
            + (f"  backlog={self.work_queue.qsize()}" if not self.work_queue.empty() else ''))

    def finalize_callback(self, request, response):
        """
        Closes out the current recording: flushes the tail window, drains the
        inference backlog, writes artifacts, and resets for the next recording.

        A service rather than a topic so the evaluation harness gets an ordering
        guarantee -- the response only comes back once this recording's files are
        on disk, which is what makes it safe to start playing the next bag.
        :param request: the Trigger request (unused)
        :param response: the Trigger response
        :return: the populated response
        """
        with self.buffer_lock:
            tail = self.windower.flush()
            duration_s = self.windower.duration_s
            audio = self.windower.full_audio() if duration_s > 0 else np.zeros(0, np.int16)
        if tail is not None:
            self.work_queue.put(tail)
        # Blocks until the worker has finished every queued window, including the
        # tail just added.
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

        # The whole-recording verdict. Runs here rather than per window because it is
        # the step that reads the encounter as a sequence -- a prompt in one window and
        # its answer in the next are only a "response" when seen together.
        recording = None
        if notes:
            try:
                recording = self.assessor.assess_recording(notes)
                self.pub_reasoning.publish(String(data=json.dumps({
                    'stamp': self.get_clock().now().nanoseconds / 1e9,
                    'recording_name': str(self.get_parameter('recording_name').value),
                    'kind': 'recording_verdict',
                    'n_windows': len(notes),
                    'notes': recording.notes,
                    'probabilities': {task.name: {c: round(float(p), 6) for c, p in
                                                  zip(task.classes,
                                                      recording.probabilities[task.name])}
                                      for task in self.tasks},
                }, sort_keys=True)))
                self.get_logger().info('Recording verdict: ' + '  '.join(
                    f'{task.name}={recording.label(task)}'
                    f'({recording.confidence(task):.2f})' for task in self.tasks))
            except Exception as exc:
                # A failed summary must not cost the per-window results already in hand.
                self.get_logger().error(f'Recording-level assessment failed: {exc}')

        message = f'{len(rows)} window(s) over {duration_s:.1f}s'
        if bool(self.get_parameter('save_artifacts').value):
            try:
                out_dir = self._write_artifacts(rows, audio, duration_s, recording)
                message += f'; artifacts in {out_dir}'
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
        """
        Clears per-recording state so nothing leaks across a recording boundary.
        :return: nothing
        """
        with self.buffer_lock:
            self.windower.reset()
            self.recording_started_at = None
        with self.rows_lock:
            self.rows = []
            self.notes = []
            self.window_traces = []
        self.max_backlog = 0

    # --------------------------------------------------------------- artifacts
    def _ground_truth(self):
        """
        Looks this recording's ground-truth attributes up in a COCO-style labels.json.

        :return: dict of attribute name -> class name, empty if unavailable
        """
        labels_path = str(self.get_parameter('labels_json').value)
        name = str(self.get_parameter('recording_name').value)
        if not labels_path or not name or not os.path.isfile(labels_path):
            return {}
        with open(labels_path) as handle:
            labels = json.load(handle)
        image_ids = {image['id'] for image in labels['images'] if image['file_name'] == name}
        for annotation in labels['annotations']:
            if annotation['image_id'] in image_ids:
                return dict(annotation.get('attributes', {}))
        return {}

    def _write_artifacts(self, rows, audio: np.ndarray, duration_s: float,
                         recording=None) -> str:
        """
        Writes this recording's per-window CSV, pooled summary, heard audio, and figures.

        :param rows: the per-window result rows
        :param audio: mono int16 at SAMPLE_RATE -- exactly the audio that was classified
        :param duration_s: length of the recording in seconds
        :param recording: the RecordingAssessment from the notes, or None if disabled
        :return: the directory everything was written to
        """
        name = str(self.get_parameter('recording_name').value) or 'recording'
        out_dir = os.path.join(str(self.get_parameter('output_dir').value) or '.', name)
        os.makedirs(out_dir, exist_ok=True)
        truth = self._ground_truth()

        windows_csv = os.path.join(out_dir, 'windows.csv')
        with open(windows_csv, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        self._write_summary(os.path.join(out_dir, 'bag_result.csv'), rows, truth,
                            name, duration_s)
        if recording is not None:
            self._write_recording_assessment(
                os.path.join(out_dir, 'recording_assessment.csv'), recording, truth, name)
            with open(os.path.join(out_dir, 'notes.txt'), 'w') as handle:
                handle.write(recording.notes)
        self._write_wav(os.path.join(out_dir, 'heard.wav'), audio)
        self._write_trace(out_dir)
        self._plot_spectrogram(os.path.join(out_dir, 'spectrogram.png'), audio, rows,
                               truth, name)
        self._plot_probability_traces(os.path.join(out_dir, 'prob_traces.png'), rows,
                                      truth, name)
        return out_dir

    def _write_trace(self, out_dir) -> None:
        """
        Writes the per-window trace: exactly the audio each window was judged on, beside
        the prompt, note and probabilities that came back from it.

        The audio is stored rather than re-derived from heard.wav and the window bounds,
        because those two agree only as long as the windowing code never changes, and
        the trace exists precisely for the times when something did change.
        :return: nothing
        """
        with self.rows_lock:
            traces = list(self.window_traces)
        if not traces:
            return
        window_dir = os.path.join(out_dir, 'trace', 'windows')
        os.makedirs(window_dir, exist_ok=True)
        for trace in traces:
            index = int(trace['window_index'])
            self._write_wav(os.path.join(window_dir, f'w{index:02d}.wav'), trace['audio'])
            with open(os.path.join(window_dir, f'w{index:02d}.json'), 'w') as handle:
                json.dump({k: v for k, v in trace.items() if k != 'audio'}, handle,
                          indent=1, sort_keys=True)

    def _write_recording_assessment(self, path, recording, truth, name) -> None:
        """
        Writes the single whole-recording verdict per task, in the same shape as
        _write_summary's rows so aggregate_metrics.py can score it as just another
        decision rule alongside the window-pooling ones.
        :return: nothing
        """
        with open(path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['task', 'pooling', 'prediction', 'confidence',
                             'ground_truth', 'correct', 'probabilities'])
            for task in self.tasks:
                probabilities = recording.probabilities[task.name]
                prediction = recording.label(task)
                gt = truth.get(task.name, '')
                writer.writerow([
                    task.name, 'summary', prediction,
                    f'{recording.confidence(task):.6f}', gt,
                    '' if not gt else str(prediction == gt),
                    ';'.join(f'{c}={p:.6f}'
                             for c, p in zip(task.classes, probabilities)),
                ])

    def _write_summary(self, path, rows, truth, name, duration_s) -> None:
        """
        Writes the recording-level result: every pooling rule's prediction side by side.

        All four pooling rules are written rather than one, because which rule to
        trust is an open question that should be answered by looking at results
        across the whole dataset -- and re-running the sweep to change it would cost
        hours. windows.csv keeps the raw per-window vectors regardless, so the
        aggregation step can still recompute any of this offline.
        :return: nothing
        """
        from microphone.gemma_audio_assessor import POOLING_METHODS

        with open(path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['recording_name', name])
            writer.writerow(['duration_s', f'{duration_s:.3f}'])
            writer.writerow(['n_windows', len(rows)])
            writer.writerow(['model_id', self.assessor.model_id])
            writer.writerow(['window_s', self.get_parameter('window_s').value])
            writer.writerow(['hop_s', self.get_parameter('hop_s').value])
            writer.writerow(['channel_mode', self.get_parameter('channel_mode').value])
            writer.writerow(['max_inference_backlog', self.max_backlog])
            writer.writerow([])
            writer.writerow(['task', 'pooling', 'prediction', 'confidence',
                             'ground_truth', 'correct', 'probabilities'])
            for task in self.tasks:
                stacked = [np.array([row[f'{task.name}_p_{c}'] for c in task.classes])
                           for row in rows]
                for method in POOLING_METHODS:
                    pooled = pool_probabilities(stacked, method)
                    prediction = task.classes[int(np.argmax(pooled))]
                    gt = truth.get(task.name, '')
                    writer.writerow([
                        task.name, method, prediction, f'{float(np.max(pooled)):.6f}', gt,
                        '' if not gt else str(prediction == gt),
                        ';'.join(f'{c}={p:.6f}' for c, p in zip(task.classes, pooled)),
                    ])

    @staticmethod
    def _write_wav(path, audio: np.ndarray) -> None:
        """
        Writes exactly the audio that was classified, so a disagreement with ground
        truth can be listened to rather than guessed at.
        :return: nothing
        """
        with wave.open(path, 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(audio.tobytes())

    def _plot_spectrogram(self, path, audio, rows, truth, name) -> None:
        """
        Spectrogram of the heard audio with per-window predictions drawn over it.
        :return: nothing
        """
        if len(audio) < 512:
            return
        freqs, times, power = spectrogram(
            audio.astype(np.float64) / 32768.0, fs=SAMPLE_RATE, nperseg=512, noverlap=384)
        decibels = 10.0 * np.log10(power + 1e-12)

        fig, axes = plt.subplots(len(self.tasks) + 1, 1, figsize=(14, 3 + 2 * len(self.tasks)),
                                 sharex=True,
                                 gridspec_kw={'height_ratios': [3] + [1] * len(self.tasks)})
        axes[0].pcolormesh(times, freqs, decibels, shading='auto', cmap='magma',
                           vmin=np.percentile(decibels, 5), vmax=np.percentile(decibels, 99.5))
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].set_title(f'{name} -- heard audio ({len(audio) / SAMPLE_RATE:.1f}s @ '
                          f'{SAMPLE_RATE} Hz mono)')

        for axis, task in zip(axes[1:], self.tasks):
            colors = plt.get_cmap('tab10')(np.linspace(0, 1, 10))
            for index, class_name in enumerate(task.classes):
                axis.plot([(r['start_s'] + r['end_s']) / 2 for r in rows],
                          [r[f'{task.name}_p_{class_name}'] for r in rows],
                          marker='o', markersize=3, color=colors[index], label=class_name)
            gt = truth.get(task.name)
            if gt in task.classes:
                axis.axhline(1.0, color=colors[task.classes.index(gt)], linestyle='--',
                             alpha=0.5, label=f'truth: {gt}')
            axis.set_ylim(-0.05, 1.05)
            axis.set_ylabel(task.name.replace('_', '\n'))
            axis.legend(loc='upper right', fontsize=7, ncol=len(task.classes) + 1)
        axes[-1].set_xlabel('Time (s)')
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

    def _plot_probability_traces(self, path, rows, truth, name) -> None:
        """
        Stacked-area view of each task's probabilities over the recording, which makes
        a label that only shows up in part of a clip obvious at a glance.
        :return: nothing
        """
        fig, axes = plt.subplots(len(self.tasks), 1, figsize=(12, 3 * len(self.tasks)),
                                 sharex=True, squeeze=False)
        centers = [(r['start_s'] + r['end_s']) / 2 for r in rows]
        for axis, task in zip(axes[:, 0], self.tasks):
            series = [[r[f'{task.name}_p_{c}'] for r in rows] for c in task.classes]
            axis.stackplot(centers, *series, labels=task.classes, alpha=0.85)
            gt = truth.get(task.name, '')
            axis.set_ylim(0, 1)
            axis.set_ylabel('probability')
            axis.set_title(f'{task.name}' + (f'   (ground truth: {gt})' if gt else ''))
            axis.legend(loc='upper right', fontsize=8, ncol=len(task.classes))
        axes[-1, 0].set_xlabel('Time (s)')
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

    def shutdown(self) -> None:
        """
        Stops the worker thread.
        :return: nothing
        """
        self.work_queue.put(_SHUTDOWN)
        self.worker.join(timeout=5.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = GemmaAudioClassificationNode()
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
