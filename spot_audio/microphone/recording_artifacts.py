#!/usr/bin/env python3

"""
Per-recording evaluation artifacts, shared by every classification node.

Both gemma_audio_classification_node.py and audio_classification_node.py write the
same files in the same schema, because the whole point of running the AST/Whisper
baseline through a node at all is to compare it against Gemma -- and a comparison
is only worth anything if the two sides are scored by identical code. Keeping this
in one module is what makes "the aggregation scripts read every algorithm through
one path" true rather than aspirational.

Files written per recording, into <output_dir>/<recording_name>/:
  windows.csv       one row per analysis window: timing, level, latency, the note or
                    transcript, and a probability column per class per task
  bag_result.csv    the recording-level result under every pooling rule, with ground
                    truth joined in when a labels.json is available
  heard.wav         exactly the audio that was classified, at SAMPLE_RATE mono
  spectrogram.png   that audio, with the per-window probabilities under it
  prob_traces.png   stacked probability traces per task

This module holds no ROS types and no model code, so it can be driven from an
offline harness as easily as from a node.
"""

import csv
import json
import os
import wave

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from microphone.gemma_audio_assessor import POOLING_METHODS, SAMPLE_RATE, pool_probabilities


def ground_truth_for(labels_json: str, recording_name: str) -> dict:
    """
    Looks a recording's attributes up in a COCO-style labels.json.

    :param labels_json: path to labels.json, or '' when unavailable
    :param recording_name: the recording's file_name in that file
    :return: attribute name -> class name, empty when unavailable
    """
    if not labels_json or not recording_name or not os.path.isfile(labels_json):
        return {}
    with open(labels_json) as handle:
        labels = json.load(handle)
    image_ids = {image['id'] for image in labels['images']
                 if image['file_name'] == recording_name}
    for annotation in labels['annotations']:
        if annotation['image_id'] in image_ids:
            return dict(annotation.get('attributes', {}))
    return {}


def window_row(window_index, start_s, end_s, audio, assessment, tasks, note_field='note'):
    """
    Builds one windows.csv row from a WindowAssessment.

    :param window_index: 0-based index of this window in the recording
    :param start_s: window start, seconds since the recording's first packet
    :param end_s: window end, same clock
    :param audio: the window as float32 in [-1, 1], for the level columns
    :param assessment: a WindowAssessment
    :param tasks: the AssessmentTasks that were scored
    :param note_field: column name for the free-text trace ('note' or 'transcript')
    :return: dict ready for csv.DictWriter
    """
    row = {
        'window_index': window_index,
        'start_s': round(start_s, 3),
        'end_s': round(end_s, 3),
        'duration_s': round(end_s - start_s, 3),
        'rms': round(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if len(audio) else 0.0, 6),
        'peak': round(float(np.max(np.abs(audio))) if len(audio) else 0.0, 6),
        'latency_s': round(sum(assessment.latency_s.values()), 4),
        # Newlines flattened so the CSV stays one physical line per window and
        # survives grep/head; the unflattened form goes to notes.txt.
        note_field: ' | '.join((assessment.note or '').splitlines()),
    }
    for task in tasks:
        probabilities = assessment.probabilities[task.name]
        for class_name, probability in zip(task.classes, probabilities):
            row[f'{task.name}_p_{class_name}'] = round(float(probability), 6)
        row[f'{task.name}_pred'] = task.classes[int(np.argmax(probabilities))]
    return row


class RecordingArtifactWriter:
    """
    Writes one recording's evaluation artifacts.
    """

    def __init__(self, tasks, output_dir, recording_name, labels_json='',
                 algorithm='unknown', metadata=None):
        """
        :param tasks: the AssessmentTasks that were scored
        :param output_dir: parent directory; a folder per recording is created inside
        :param recording_name: names the folder and joins to ground truth
        :param labels_json: optional COCO-style labels.json for ground truth
        :param algorithm: free-text label recorded in bag_result.csv, so a results
                          folder is self-describing rather than identified by path
        :param metadata: extra key/value rows for bag_result.csv (window size, model
                         id, backlog...) -- whatever makes the run reproducible
        """
        self.tasks = tuple(tasks)
        self.output_dir = output_dir
        self.recording_name = recording_name or 'recording'
        self.labels_json = labels_json
        self.algorithm = algorithm
        self.metadata = dict(metadata or {})

    def write(self, rows, audio, duration_s, notes_text=''):
        """
        Writes every artifact for one recording.

        :param rows: per-window rows from window_row()
        :param audio: mono int16 at SAMPLE_RATE -- exactly what was classified
        :param duration_s: recording length in seconds
        :param notes_text: full unflattened reasoning trace, if any
        :return: the directory written to
        """
        directory = os.path.join(self.output_dir or '.', self.recording_name)
        os.makedirs(directory, exist_ok=True)
        truth = ground_truth_for(self.labels_json, self.recording_name)

        with open(os.path.join(directory, 'windows.csv'), 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        self._write_summary(os.path.join(directory, 'bag_result.csv'), rows, truth,
                            duration_s)
        if notes_text:
            with open(os.path.join(directory, 'notes.txt'), 'w') as handle:
                handle.write(notes_text)
        self._write_wav(os.path.join(directory, 'heard.wav'), audio)
        self._plot_spectrogram(os.path.join(directory, 'spectrogram.png'), audio, rows,
                               truth)
        self._plot_traces(os.path.join(directory, 'prob_traces.png'), rows, truth)
        return directory

    def _write_summary(self, path, rows, truth, duration_s):
        """
        Writes the recording-level result under every pooling rule.

        All rules are written rather than one, because which to trust is a question
        answered across the whole data set, and re-running a sweep to change it costs
        hours. windows.csv keeps the raw vectors either way.
        :return: nothing
        """
        with open(path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['recording_name', self.recording_name])
            writer.writerow(['algorithm', self.algorithm])
            writer.writerow(['duration_s', f'{duration_s:.3f}'])
            writer.writerow(['n_windows', len(rows)])
            for key, value in self.metadata.items():
                writer.writerow([key, value])
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
    def _write_wav(path, audio):
        with wave.open(path, 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(audio.tobytes())

    def _plot_spectrogram(self, path, audio, rows, truth):
        """
        Spectrogram of the heard audio with the per-window probabilities beneath it.
        :return: nothing
        """
        if len(audio) < 512:
            return
        frequencies, times, power = spectrogram(
            audio.astype(np.float64) / 32768.0, fs=SAMPLE_RATE, nperseg=512, noverlap=384)
        decibels = 10.0 * np.log10(power + 1e-12)

        figure, axes = plt.subplots(
            len(self.tasks) + 1, 1, figsize=(14, 3 + 2 * len(self.tasks)), sharex=True,
            gridspec_kw={'height_ratios': [3] + [1] * len(self.tasks)})
        axes[0].pcolormesh(times, frequencies, decibels, shading='auto', cmap='magma',
                           vmin=np.percentile(decibels, 5),
                           vmax=np.percentile(decibels, 99.5))
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].set_title(f'{self.recording_name} — {self.algorithm} — '
                          f'{len(audio) / SAMPLE_RATE:.1f}s @ {SAMPLE_RATE} Hz mono')

        centers = [(r['start_s'] + r['end_s']) / 2 for r in rows]
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, 10))
        for axis, task in zip(axes[1:], self.tasks):
            for index, class_name in enumerate(task.classes):
                axis.plot(centers, [r[f'{task.name}_p_{class_name}'] for r in rows],
                          marker='o', markersize=3, color=colors[index], label=class_name)
            gt = truth.get(task.name)
            if gt in task.classes:
                axis.axhline(1.0, color=colors[task.classes.index(gt)], linestyle='--',
                             alpha=0.5, label=f'truth: {gt}')
            axis.set_ylim(-0.05, 1.05)
            axis.set_ylabel(task.name.replace('_', '\n'))
            axis.legend(loc='upper right', fontsize=7, ncol=len(task.classes) + 1)
        axes[-1].set_xlabel('Time (s)')
        figure.tight_layout()
        figure.savefig(path, dpi=110)
        plt.close(figure)

    def _plot_traces(self, path, rows, truth):
        """
        Stacked probability traces, which make a label present in only part of a
        recording obvious at a glance.
        :return: nothing
        """
        figure, axes = plt.subplots(len(self.tasks), 1,
                                    figsize=(12, 3 * len(self.tasks)), sharex=True,
                                    squeeze=False)
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
        figure.suptitle(f'{self.recording_name} — {self.algorithm}')
        figure.tight_layout()
        figure.savefig(path, dpi=110)
        plt.close(figure)
