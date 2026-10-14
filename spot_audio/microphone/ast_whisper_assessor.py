#!/usr/bin/env python3

"""
The AST + Whisper baseline, wrapped in the same interface as GemmaAudioAssessor so
both can be scored by one harness on identical windows.

This exists to answer "is the unified model actually better than what we had", and
that question is only meaningful if the comparison is like-for-like: same 30 s
windows, same pooling, same metrics, same ground truth. So this module deliberately
produces the SAME shape of output -- a probability vector per task per window --
even though neither underlying model natively emits one.

TWO MODELS, TWO SEPARATE ENVIRONMENTS
Both imports are lazy and independent, because on this machine they cannot coexist:
AST needs torch + transformers (/home/cdcl/.venv), while Whisper is a CTranslate2
model that needs no torch at all and lives in its own venv with its own CUDA libs
(roboscout-assessment/audio_classification/venv_whisper). `backends` selects which
half to load, so the same file runs under either interpreter and a missing package
is never an error until something actually asks for that backend.

WHAT EACH MODEL CAN AND CANNOT ANSWER
  AST      both labels. It is an AudioSet tagger, so its 527 sigmoid outputs have to
           be grouped into the triage classes -- see the label sets below, which are
           the honest part of this file and where its ceiling really lives.
  Whisper  verbal alertness only. It is a transcriber; it has no notion of
           respiratory sounds, so claiming a respiratory-distress number from it
           would be inventing one. supported_tasks() says so explicitly rather than
           emitting a uniform vector that would silently score as a real prediction.
"""

from dataclasses import dataclass, field
import numpy as np
import time
from typing import Dict, List, Optional, Sequence, Tuple

from microphone.gemma_audio_assessor import (
    ALERTNESS_VERBAL_TASK, AssessmentTask, RESPIRATORY_DISTRESS_TASK, SAMPLE_RATE,
    WindowAssessment)

# AST's feature extractor pads or truncates to max_length=1024 frames, which at a
# 10 ms hop is exactly 10.24 s. A 30 s window therefore cannot be classified in one
# pass; it is split into chunks of this length and the per-chunk probabilities pooled.
#
# MEASURED, and the reason this is 10.24 rather than a round 10: the extractor pads the
# FILTERBANK, not the waveform, and it pads with zeros BEFORE normalization, so padded
# frames read as fairly loud audio rather than silence. On a clip whose full window
# scores Speech 0.44, a 3 s piece scores 0.21 and a 1 s piece 0.075 -- and padding the
# waveform with silence first is worse still (0.055 and 0.026), because the model then
# sees mostly silence. Short pieces are therefore never scored: the final chunk slides
# back to cover the end of the audio instead. 10.24 s leaves no padding at all.
AST_CHUNK_SECONDS = 10.24

# ---------------------------------------------------------------------------
# AudioSet label groups. Indices are AST's own ids (see the checkpoint's config.json).
#
# RESPIRATORY DISTRESS, restricted to the audible signs the assessment protocol
# actually names -- gasping, snoring, wheezing, rapid shallow breathing, grunting
# on exhalation:
#     39 Grunt   42 Wheeze   43 Snoring   44 Gasp   45 Pant   46 Snort
#
# This deliberately does NOT reproduce the group in audio_classifier.py, which was
# [43, 45, 46, 48, 49, 42] -- that set includes 48 "Throat clearing" and 49
# "Sneeze", neither of which is a sign of respiratory distress under any reading of
# the protocol, and both of which are common in ordinary recordings. It also leaves
# out 44 "Gasp", which the protocol names explicitly. Keeping the old grouping would
# have measured a bug rather than the model.
#
# 41 "Breathing" is excluded on purpose: audible breathing is not distressed
# breathing, and conflating the two is exactly the failure mode already measured in
# the Gemma run.
AUDIOSET_RESPIRATORY_DISTRESS = (39, 42, 43, 44, 45, 46)

# VERBAL ALERTNESS. 'normal' is coherent intelligible speech only:
#     0 Speech  1 Male speech  2 Female speech  3 Child speech
#     4 Conversation  5 Narration/monologue
# Excluded from 'normal' on purpose:
#   6  Babbling           -- vocal but not coherent, so it belongs in abnormal
#   7  Speech synthesizer -- this is the ROBOT. Counting it as the casualty
#                            speaking is the single largest error source measured
#                            in this task, and here it can be excluded by id.
#   68 Chatter, 69 Crowd, 70 Hubbub/speech babble -- bystanders and background,
#                            never the casualty.
AUDIOSET_VERBAL_NORMAL = (0, 1, 2, 3, 4, 5)

# 'abnormal' is any human vocalization that is not coherent speech: shouting,
# screaming, crying, moaning, whimpering, groaning, grunting, gasping, coughing,
# babbling, whispering, sighing, humming, and laughter. Laughter is a poor fit for
# a protocol phrased around pain and distress, but it is a vocalization and the
# alternative bucket is 'absent', which would be plainly wrong.
AUDIOSET_VERBAL_ABNORMAL = (6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                            22, 23, 24, 25, 26, 37, 38, 39, 44, 47)

# Whisper's avg_logprob is roughly -0.1 for a clean confident transcription and
# drops below -1.0 when it is guessing at noise. This maps that range onto a
# coherence score in (0, 1): the midpoint sits at MIDPOINT and the transition is
# gradual rather than a hard threshold, so the resulting probability vector carries
# the model's uncertainty instead of discarding it.
WHISPER_LOGPROB_MIDPOINT = -0.6
WHISPER_LOGPROB_SCALE = 0.35


def noisy_or(probabilities: np.ndarray, indices: Sequence[int]) -> float:
    """
    P(at least one of these labels is active), treating them as independent.

    :param probabilities: per-label sigmoid outputs
    :param indices: the label ids forming this group
    :return: probability that any of them fired
    """
    if not indices:
        return 0.0
    return float(1.0 - np.prod(1.0 - probabilities[list(indices)]))


def none_of(probabilities: np.ndarray, indices: Sequence[int]) -> float:
    """
    :return: probability that none of these labels fired
    """
    if not indices:
        return 1.0
    return float(np.prod(1.0 - probabilities[list(indices)]))


class AstWhisperAssessor:
    """
    Scores windows with AST and/or Whisper, emitting GemmaAudioAssessor-shaped output.
    """

    def __init__(self,
                 ast_path: str = '/home/cdcl/cdcl_ws/models/alertness_verbal_classifiers/',
                 whisper_path: str = '/home/cdcl/cdcl_ws/models/whisper/',
                 backends: Sequence[str] = ('ast',),
                 device: str = 'cuda',
                 whisper_vad: bool = True,
                 logger=None):
        """
        :param ast_path: directory holding the AST checkpoint and preprocessor config
        :param whisper_path: directory holding the CTranslate2 Whisper model
        :param backends: any of 'ast', 'whisper' -- which halves to load
        :param device: torch / CTranslate2 device
        :param whisper_vad: Whisper's Silero VAD pre-filter. The deployed node runs
                            with it ON, which is the honest default to measure; it
                            also rejects most of this data set outright, so
                            score_ast_whisper_baselines.py can sweep it.
        :param logger: anything with .info/.warning/.error; optional
        """
        self.ast_path = ast_path
        self.whisper_path = whisper_path
        self.backends = tuple(backends)
        self.device = device
        self.whisper_vad = whisper_vad
        self.logger = logger
        self.ast_model = None
        self.ast_feature_extractor = None
        self.whisper_model = None

    def _log(self, level: str, message: str) -> None:
        if self.logger is not None:
            getattr(self.logger, level)(message)

    def supported_tasks(self) -> Tuple[AssessmentTask, ...]:
        """
        :return: the tasks these backends can actually answer. Whisper alone cannot
                 speak to respiratory distress, and saying so beats emitting a
                 placeholder vector that would be scored as if it were a prediction.
        """
        if 'ast' in self.backends:
            return (RESPIRATORY_DISTRESS_TASK, ALERTNESS_VERBAL_TASK)
        return (ALERTNESS_VERBAL_TASK,)

    def load(self) -> None:
        """
        Loads whichever backends were requested. Blocking; call once.
        :return: nothing
        """
        if 'ast' in self.backends:
            import os
            import torch
            from transformers import AutoFeatureExtractor, ASTForAudioClassification

            start = time.time()
            self.ast_feature_extractor = AutoFeatureExtractor.from_pretrained(
                os.path.join(self.ast_path, 'preprocessor_config.json'))
            self.ast_model = ASTForAudioClassification.from_pretrained(
                self.ast_path).to(self.device).eval()
            self._log('info', f'Loaded AST from {self.ast_path} '
                              f'({self.ast_model.config.num_labels} labels) '
                              f'in {time.time() - start:.1f}s')

        if 'whisper' in self.backends:
            from faster_whisper import WhisperModel

            start = time.time()
            self.whisper_model = WhisperModel(
                self.whisper_path, device=self.device,
                compute_type='float16' if self.device == 'cuda' else 'int8',
                local_files_only=True)
            self._log('info', f'Loaded Whisper from {self.whisper_path} '
                              f'in {time.time() - start:.1f}s')

    # ----------------------------------------------------------------- AST
    def ast_label_probabilities(self, audio: np.ndarray) -> np.ndarray:
        """
        Runs AST over the window and pools its per-label probabilities.

        The window is split into AST_CHUNK_SECONDS pieces because the feature
        extractor silently truncates anything longer, and pooled with a MAXIMUM
        rather than a mean: these are detections of transient events -- a single
        gasp, one word -- and averaging over three chunks would dilute a sound
        present in only one of them, which is the opposite of what a detector
        should do.
        :param audio: mono float32 in [-1, 1] at SAMPLE_RATE
        :return: (527,) array of per-label probabilities
        """
        import torch

        chunk = int(AST_CHUNK_SECONDS * SAMPLE_RATE)
        starts = list(range(0, max(len(audio) - chunk, 0) + 1, chunk))
        # The tail is covered by sliding a full chunk back over the end rather than
        # scoring a short piece, which is mostly padding -- see AST_CHUNK_SECONDS.
        if len(audio) > chunk and starts[-1] + chunk < len(audio):
            starts.append(len(audio) - chunk)
        pieces = [audio[start:start + chunk] for start in starts] or [audio]

        pooled = None
        for piece in pieces:
            inputs = self.ast_feature_extractor(
                piece, sampling_rate=SAMPLE_RATE, return_tensors='pt')
            with torch.no_grad():
                logits = self.ast_model(inputs['input_values'].to(self.device)).logits
            # AudioSet is multi-label, so sigmoid per label -- not softmax.
            probabilities = torch.sigmoid(logits)[0].float().cpu().numpy()
            pooled = probabilities if pooled is None else np.maximum(pooled, probabilities)
        return pooled

    def ast_segment_probabilities(self, audio: np.ndarray,
                                  chunk_s: float = AST_CHUNK_SECONDS,
                                  hop_s: float = AST_CHUNK_SECONDS
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs AST on successive chunks of a recording and keeps every chunk's scores
        instead of pooling them, so each can be lined up with the time it came from --
        which is what putting AST's findings into a Gemma prompt, or onto a figure,
        needs.

        AST was trained on 10-second AudioSet clips (1024 frames at a 10 ms hop), so
        chunk_s defaults to that; a shorter chunk is padded by the feature extractor.
        :param audio: mono float32 in [-1, 1] at SAMPLE_RATE
        :param chunk_s: seconds of audio per AST pass
        :param hop_s: seconds between chunk starts; less than chunk_s overlaps them
        :return: (chunk start times in seconds (n,), per-label sigmoid scores (n, 527)).
                 The last chunk is slid back so the end of the audio is covered by a
                 full-length chunk. Audio shorter than chunk_s is scored in one padded
                 pass, which the model handles poorly -- see AST_CHUNK_SECONDS.
        """
        import torch

        chunk, hop = int(chunk_s * SAMPLE_RATE), int(hop_s * SAMPLE_RATE)
        starts = list(range(0, max(len(audio) - chunk, 0) + 1, hop))
        # Cover the end with a FULL chunk slid back, never a short one -- a short piece
        # is mostly padding and the model scores it as something else entirely.
        if len(audio) > chunk and starts[-1] + chunk < len(audio):
            starts.append(len(audio) - chunk)

        scores = []
        for start in starts:
            inputs = self.ast_feature_extractor(
                audio[start:start + chunk], sampling_rate=SAMPLE_RATE, return_tensors='pt')
            with torch.no_grad():
                logits = self.ast_model(inputs['input_values'].to(self.device)).logits
            # AudioSet is multi-label, so sigmoid per label -- not softmax.
            scores.append(torch.sigmoid(logits)[0].float().cpu().numpy())
        return np.array(starts, dtype=np.float64) / SAMPLE_RATE, np.array(scores)

    @staticmethod
    def ast_to_tasks(label_probabilities: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Groups AudioSet label probabilities into the two triage distributions.

        :param label_probabilities: (527,) sigmoid outputs
        :return: task name -> probability vector in that task's class order
        """
        present = noisy_or(label_probabilities, AUDIOSET_RESPIRATORY_DISTRESS)
        respiratory = np.array([1.0 - present, present], dtype=np.float64)

        normal = noisy_or(label_probabilities, AUDIOSET_VERBAL_NORMAL)
        abnormal = noisy_or(label_probabilities, AUDIOSET_VERBAL_ABNORMAL)
        # 'absent' is the probability that NO vocalization label fired at all, which
        # is what the class actually means -- not a leftover bucket.
        absent = none_of(label_probabilities,
                         tuple(AUDIOSET_VERBAL_NORMAL) + tuple(AUDIOSET_VERBAL_ABNORMAL))
        verbal = np.array([normal, abnormal, absent], dtype=np.float64)
        verbal = verbal / max(verbal.sum(), 1e-12)
        return {RESPIRATORY_DISTRESS_TASK.name: respiratory,
                ALERTNESS_VERBAL_TASK.name: verbal}

    # ------------------------------------------------------------- Whisper
    def whisper_to_verbal(self, audio: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Turns a transcription into a verbal-alertness distribution.

        Whisper does not classify anything, so this is a stated rule, not a learned
        one, and it mirrors the same decomposition the symptom questions use:

            P(absent)   = 1 - P(speech)
            P(normal)   = P(speech) * coherence
            P(abnormal) = P(speech) * (1 - coherence)

        P(speech) is the best segment's 1 - no_speech_prob, and coherence comes from
        avg_logprob -- Whisper's own confidence in what it wrote, which is the only
        signal it offers about whether speech was intelligible or garbled.

        The load-bearing caveat is that Whisper transcribes EVERY voice. On this data
        it returns the responder's words ("There we go. Thank you.") on recordings
        where the casualty never speaks, so a confident transcription is not evidence
        the casualty said anything. Nothing in this mapping can fix that; it is a
        property of the model and belongs in the interpretation of the result.
        :param audio: mono float32 in [-1, 1] at SAMPLE_RATE
        :return: (probability vector over normal/abnormal/absent, transcript)
        """
        segments, _ = self.whisper_model.transcribe(
            audio, language='en', task='transcribe',
            vad_filter=self.whisper_vad, condition_on_previous_text=False)
        segments = list(segments)
        if not segments:
            return np.array([0.0, 0.0, 1.0]), ''

        speech = max(1.0 - float(s.no_speech_prob) for s in segments)
        mean_logprob = float(np.mean([s.avg_logprob for s in segments]))
        coherence = 1.0 / (1.0 + np.exp(
            -(mean_logprob - WHISPER_LOGPROB_MIDPOINT) / WHISPER_LOGPROB_SCALE))
        vector = np.array([speech * coherence, speech * (1.0 - coherence), 1.0 - speech])
        return vector / max(vector.sum(), 1e-12), ' '.join(s.text.strip() for s in segments)

    # -------------------------------------------------------------- driver
    def assess_window(self, audio: np.ndarray, **_) -> WindowAssessment:
        """
        Scores one window with whichever backends are loaded.

        Accepts and ignores GemmaAudioAssessor's keyword arguments (take_notes,
        ground_in_note, robot_speech) so one harness can drive either assessor
        without special-casing.
        :param audio: mono float32 in [-1, 1] at SAMPLE_RATE
        :return: probabilities per supported task, latencies, and any transcript
        """
        result = WindowAssessment()
        if 'ast' in self.backends:
            start = time.time()
            grouped = self.ast_to_tasks(self.ast_label_probabilities(audio))
            elapsed = time.time() - start
            for task in (RESPIRATORY_DISTRESS_TASK, ALERTNESS_VERBAL_TASK):
                result.probabilities[task.name] = grouped[task.name]
                result.latency_s[task.name] = elapsed / 2.0
        if 'whisper' in self.backends:
            start = time.time()
            vector, transcript = self.whisper_to_verbal(audio)
            # Whisper overrides AST on verbal alertness when both are loaded; the
            # scorer normally runs them separately so each gets its own numbers.
            result.probabilities[ALERTNESS_VERBAL_TASK.name] = vector
            result.latency_s['whisper'] = time.time() - start
            result.note = transcript
        return result
