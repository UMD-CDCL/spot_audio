#!/usr/bin/env python3

"""
Casualty audio assessment with a single unified Gemma 4 model, replacing the
AST + Whisper pair in audio_classifier.py / transcriber.py.

Two independent labels are produced per audio window:
  respiratory_distress -- absent | present
  alertness_verbal     -- normal | abnormal | absent

Both come out as CALIBRATABLE PROBABILITY VECTORS, not just argmax labels, which
is the whole reason this scores logits instead of parsing generated text. The
model is asked a multiple-choice question whose answer is forced to begin at a
known position (see PREFILL), and the probability of each class is read directly
off the softmax over that one position's logits, restricted to the class tokens.
Free-text generation would give a label but no usable confidence, and the
downstream evaluation (Brier score, ECE, reliability diagrams) needs real
probabilities to mean anything.

Every class string is asserted to be exactly one token at load time -- see
_resolve_class_tokens(). That's what makes the single-position softmax exact
rather than an approximation over first tokens of multi-token words.

Audio contract: mono float32 in [-1, 1] at SAMPLE_RATE (16 kHz). The model caps a
single audio segment at MAX_AUDIO_SECONDS (audio_seq_length 750 tokens x
audio_ms_per_token 40 ms), so callers must window longer recordings.

This module deliberately holds no ROS types: gemma_audio_classification_node.py
wraps it for live use, and the offline evaluation harness imports the same class
so both paths score audio identically.
"""

from dataclasses import dataclass, field
import numpy as np
import time
from typing import Dict, List, Optional, Sequence, Tuple

# 16 kHz mono is what Gemma 4's audio feature extractor expects.
SAMPLE_RATE = 16000

# audio_seq_length (750 soft tokens) * audio_ms_per_token (40 ms). Audio longer
# than this in a single segment is silently truncated by the processor, so
# callers window rather than let that happen invisibly.
MAX_AUDIO_SECONDS = 30.0

# The answer is forced to start right after this prefilled assistant text, so the
# class token always lands at one known position. It also makes every class a
# SPACE-PREFIXED word (" absent", " normal", ...), which is what makes them
# single tokens in Gemma's tokenizer -- bare "absent" tokenizes as ("abs", "ent")
# and "abnormal" as ("ab", "normal"), which would make a single-position softmax
# wrong rather than merely imprecise.
PREFILL = 'Answer:'

# Shared preamble. The recordings contain responders and bystanders talking over
# the casualty, and the ground-truth labels describe the CASUALTY only, so this
# says so explicitly -- without it the model happily reports a medic's calm,
# fluent speech as the casualty's own verbal alertness.
_CONTEXT = (
    "You are listening to a recording captured by a robot assessing a single injured "
    "casualty at an emergency scene. Other people (responders, bystanders) may also be "
    "audible, along with background and machine noise. Judge only the injured casualty, "
    "not anyone else, and not the robot."
)


# Asked of every window when note-taking is on. Its job is OBSERVATION, not
# judgement: notes record what the casualty produced, window by window, so that
# assess_recording() can apply the rubric once over the whole timeline. Everything is
# asked for literally, because the moment this step starts diagnosing, the final step
# is reasoning about its guesses rather than about the audio.
#
# VERBAL ALERTNESS ONLY. The previous version also carried a BREATHING checklist for
# respiratory distress and a PROMPTS heading with a worked example. MEASURED on seven
# hand-picked 30 s clips, it wrote the same template for digital silence as for a
# casualty yelling "Help! Help! I can't move my leg!" -- "2 prompts, e.g. 'Can you hear
# me?'", breathing "(a) yes (b) laboured (c) 4", CASUALTY SPEECH: none -- and across the
# 96 independent recordings it reported casualty speech in 8% of windows, against 25%
# before the checklist was added. Respiratory distress is not this module's job (AST
# handles it in audio_classification_node.py), so the checklist was deleted rather than
# kept as an option.
#
# This three-heading form quoted that casualty verbatim and wrote 'none' for silence,
# but over whole recordings it has the opposite habit: on the 28 truly silent
# recordings it wrote a casualty sound -- usually "The casualty makes a low, groaning
# sound." -- in 111 of 232 windows, where AST scores every groan, moan and cry label
# at 0.000. It is the baseline for note-prompt iteration, not a finished prompt.
#
# NO CANDIDATE WORD LISTS under the casualty headings: spelling out "moaning, groaning,
# screaming, crying" made greedy decoding echo the list back verbatim, which looks like
# an observation and carries no information. The rubric still enumerates, because
# there the words are applied rather than transcribed.
#
# OTHER VOICES comes first so that speech from the robot and responders has somewhere
# to go other than CASUALTY SPEECH.
NOTE_PROMPT = (
    f"{_CONTEXT}\n\n"
    "Write brief factual notes on what is audible in this segment, under exactly these "
    "three headings:\n"
    "OTHER VOICES: in one short line, say whether a robot or responders are talking, and "
    "roughly how much.\n"
    "CASUALTY SPEECH: quote verbatim any words the injured casualty themself says. Write "
    "'none' if the casualty says no words.\n"
    "CASUALTY SOUNDS: describe in your own words any non-speech vocal sound the casualty "
    "makes. Write 'none' if there is none.\n\n"
    "Report only what you can actually hear in THIS segment."
)

# Hard backstop against degenerate repetition. Greedy decoding on audio where a robot
# repeats "Can you hear me?" every few seconds can lock into re-emitting one phrase
# until max_new_tokens, truncating the headings after it; blocking any verbatim
# 16-token span forces it on. 16 is long enough that ordinary prose never trips it.
NOTE_NO_REPEAT_NGRAM = 16


@dataclass(frozen=True)
class AssessmentTask:
    """
    One independent classification, asked either of a single window's audio or of a
    whole recording's accumulated notes.

    :param name: column/key name used in outputs (e.g. 'respiratory_distress')
    :param classes: ordered class names; index order is the probability vector order
                    and must match the evaluation's expected ordering
    :param rubric: the class definitions, shared by both prompts so the per-window
                   and whole-recording answers are decided against identical criteria
                   and are therefore comparable
    :param window_caveat: what to tell the model about judging a 30 s excerpt alone
    :param observation_module: ObservationModule constant published for this task
    :param summary_instruction: how to combine the segments into one verdict; sits
                                between the rubric and the answer instruction in
                                summary_prompt()
    """
    name: str
    classes: Tuple[str, ...]
    rubric: str
    window_caveat: str
    observation_module: str
    summary_instruction: str = (
        "Apply the definitions to the encounter as a whole, counting prompts and "
        "responses across all segments rather than within any single one.")

    def _instruction(self) -> str:
        options = ', '.join(f"'{c}'" for c in self.classes)
        return f"Reply with exactly one word, one of: {options}."

    def answer_instruction(self) -> str:
        """
        :return: the closing line that forces a one-word answer, for callers building a
                 prompt of their own around this task's rubric
        """
        return self._instruction()

    def window_prompt(self, note: Optional[str] = None,
                      robot_speech: Optional[Sequence[str]] = None) -> str:
        """
        Builds the prompt for scoring one window, optionally grounded in evidence
        gathered before the question is asked.

        :param note: this window's observation notes (see NOTE_PROMPT). Supplying
                     them makes the model commit IN WRITING to who said what before
                     it is asked to judge -- which is a different intervention from
                     telling it whom to ignore, and the only one with direct
                     evidence of working: the notes already separate PROMPTS from
                     CASUALTY SPEECH correctly ('CASUALTY SPEECH: "Yeah." (answers
                     prompt)') while both prompt-only regimes mis-attributed.
        :param robot_speech: what the robot itself said during this window. On the
                             real platform this is known exactly rather than
                             inferred -- see PlaySound.srv, which carries the text
                             and its start/end time -- so the single largest source
                             of confusable speech can be removed by fiat instead of
                             by asking the model to recognize a synthetic voice.
        :return: the prompt text
        """
        blocks = [_CONTEXT]
        if robot_speech:
            quoted = '\n'.join(f'  - "{utterance}"' for utterance in robot_speech)
            blocks.append(
                "THE ROBOT ITSELF SPOKE THESE EXACT WORDS during this segment:\n"
                f"{quoted}\n"
                "This is verified, not a guess. Any voice saying these words is the "
                "robot. It is never the casualty.")
        if note:
            blocks.append(
                "OBSERVATION NOTES already taken for this segment:\n"
                f"{note}\n"
                "Use these notes as your record of who said what. If they attribute "
                "something to a responder or the robot, it is not the casualty.")
        blocks += [self.rubric, self.window_caveat, self._instruction()]
        return '\n\n'.join(blocks)

    def summary_prompt(self, notes: str) -> str:
        """
        :return: the prompt used to score the whole recording from its window notes
        """
        return (f"{_CONTEXT}\n\n"
                f"Below are observation notes taken from consecutive segments of one "
                f"recording, in time order. Together they cover the whole encounter.\n\n"
                f"{notes}\n\n{self.rubric}\n\n"
                f"{self.summary_instruction}\n\n"
                f"{self._instruction()}")


# MEASURED RESULT OF THE SIGN-DETECTION REFRAMING BELOW -- read before reusing it.
# Reframing this task from diagnosis to sign detection (and adding the gated breathing
# checklist in NOTE_PROMPT) was tested head-to-head against the previous diagnosis
# framing on a balanced 40-recording subset, 20 per class, all longer than 60 s:
#
#                            accuracy   kappa   predicted 'present'
#   diagnosis framing (old)     0.550  +0.100         22 / 40
#   sign detection    (new)     0.500   0.000         40 / 40
#
# The new framing is better motivated -- it matches the written criteria and stops
# asking the model to infer a partly visual diagnosis -- and it scored WORSE, by
# collapsing to 'present' for every recording.
#
# The mechanism is the interesting part. Asking "can you hear the casualty breathing
# at all? yes/no" moved the reported audibility rate from 10.6% of segments to 94.3%:
# the model largely acquiesces to a direct yes/no question. But the named signs then
# came out at 129 segments for ground-truth 'absent' against 128 for 'present' --
# indistinguishable, exactly as the 9.5%/11.4% split was under the old framing.
#
# So two prompt regimes that differ by ~84 points in how often they claim to hear
# breathing produce observations with the same (zero) mutual information with the
# label. That is much stronger evidence than either run alone that the signal is not
# in this audio, rather than that the prompt was wrong. Further prompt iteration on
# this task is not indicated; confirming how the labels were annotated is.
RESPIRATORY_DISTRESS_TASK = AssessmentTask(
    name='respiratory_distress',
    # Order matches RD_ORDER in roboscout-assessment's plot_audio_dataset_stats.py
    # and the dataset's own attribute values.
    classes=('absent', 'present'),
    # Phrased as sign detection, not as a diagnosis. The assessment protocol's full
    # definition is partly VISUAL -- tripod position, open mouth, abnormal head/neck
    # position, involuntary inward curling of the hands -- and three of its four
    # clauses pair one of those with a sound. None of that is available here, so asking
    # the model to decide "is this person in respiratory distress" asks it to infer
    # across a modality it cannot observe, and it answered by proxy: it equated "I can
    # hear breathing at all" with distress, which is uncorrelated with the label.
    # What IS decidable from audio is the sign list the protocol names -- gasping,
    # snoring, wheezing, rapid shallow breathing, and a respiratory rate over 35 BPM --
    # so the rubric asks for exactly those and says plainly what it cannot see.
    rubric=(
        "Decide whether the casualty shows AUDIBLE signs of respiratory distress.\n"
        "present: you can hear at least one of -- gasping or intermittent gasping; "
        "snoring or snorting respiration; wheezing or stridor; rapid shallow breathing; "
        "laboured, obstructed, gurgling or rattling breathing; or a respiratory rate "
        "faster than about 35 breaths per minute (more than roughly 17 breaths in a "
        "30-second segment).\n"
        "absent: you can hear the casualty breathing and it is quiet, unlaboured and at "
        "an ordinary rate; OR you cannot hear the casualty breathing at all.\n"
        "Not hearing any breathing is 'absent', never 'present' -- silence is not "
        "evidence of distress. Equally, hearing breathing is not by itself evidence of "
        "distress; one of the specific signs above must be audible.\n"
        "The full clinical definition also includes visual signs -- tripod posture, open "
        "mouth, abnormal head or neck position, inward curling of the hands. You cannot "
        "see the casualty, so judge only on the sounds listed above and do not guess at "
        "posture or appearance."
    ),
    window_caveat=(
        "You are hearing one segment of a longer recording. Judge only this segment."
    ),
    observation_module='gemma_audio_respiratory_distress',
)

ALERTNESS_VERBAL_TASK = AssessmentTask(
    name='alertness_verbal',
    # Order matches the dataset's categories list in labels.json
    # (id 0 normal, 1 abnormal, 2 absent) and VERBAL_ORDER in the stats plotter.
    classes=('normal', 'abnormal', 'absent'),
    # PROMPT-INDEPENDENT BY DESIGN. An earlier version of this rubric was phrased
    # around the protocol's "responsive to prompts after at most 2 attempts", which
    # made every judgement conditional on hearing a questioner. That is the wrong
    # dependency for this module: recordings are coming where nobody asks the
    # casualty anything, and the final test set has a robot asking instead of a
    # person. The judgement has to survive both.
    #
    # It is also unnecessary. Deciding mental state does not require knowing the
    # right answer to the question -- a sensible utterance is sensible on its own
    # terms, and a non-sequitur is recognizable as one without knowing what was
    # asked. So the rubric below turns on WHAT THE CASUALTY PRODUCES, never on
    # whether someone prompted them.
    rubric=(
        "Rate the casualty's verbal alertness -- how intact their mental state sounds.\n"
        "normal: the casualty says something sensible. Coherent, intelligible words "
        "that hang together -- either a relevant reply to someone, or a sensible "
        "statement of their own such as naming a problem, asking for help, or "
        "describing where they hurt. It does not matter whether anyone asked them "
        "anything, and it does not matter whether you know what was asked; sensible "
        "speech is recognizable on its own.\n"
        "abnormal: the casualty makes sound but it is not sensible speech. Screaming "
        "or moaning in pain, crying, groaning, whimpering, grunting; or speech that "
        "is slurred, mumbled, rambling, confused, or a non-sequitur that does not "
        "hang together. A casualty who shouts in agony instead of answering is "
        "abnormal, not normal. "
        # Decided by the user 2026-09-15 after reviewing recordings labelled abnormal
        # in which the casualty answers orientation questions with "I don't know"
        # over and over: perseveration is abnormal. Phrased around the repetition,
        # not the questions, so it holds whether or not anyone asks anything.
        "So is a casualty who says the same thing over and over -- for example "
        "'I don't know, I don't know, I don't know' -- even though the phrase would "
        "be sensible said once.\n"
        "absent: the casualty makes no vocal sound of any kind.\n"
        "Only the injured casualty's own voice counts. Responders, medics, "
        "bystanders and the robot are not the casualty, however clearly you hear "
        "them -- a scene full of other people talking while the casualty stays "
        "silent is 'absent'."
    ),
    window_caveat=(
        "You are hearing one segment of a longer recording. Judge this segment on "
        "the sounds the casualty makes in it."
    ),
    observation_module='gemma_audio_alertness_verbal',
    # The default instruction tells the verdict step to count "prompts and responses",
    # which contradicts the prompt-independent rubric above and leaves nothing to count
    # once the notes stop recording prompts (VERBAL_NOTE_PROMPT). This one asks only
    # what the rubric asks: what the casualty produced, anywhere in the encounter.
    summary_instruction=(
        "Apply the definitions to the encounter as a whole, using everything the "
        "casualty says or vocalizes in any segment. Segments in which the casualty is "
        "silent do not make the encounter 'absent' when the casualty is heard in "
        "another segment."),
)

# MINIMAL variant: the same three classes with the attribution language REMOVED.
#
# Every rubric revision so far has added more insistence that other voices be
# ignored, and each one pushed the model further toward 'absent' -- measured:
# 'absent' recall went 0.750 -> 1.000 while 'abnormal' fell 0.553 -> 0.128 and
# 'normal' was never predicted at all. The model cannot reliably tell who is
# speaking, so an instruction to attribute resolves, every time, as "not the
# casualty", and 'absent' swallows the data set.
#
# This variant tests the opposite hypothesis: describe the SCENE acoustically and
# let the class fall out of what kind of vocalization dominates. It gives up on
# excluding responders -- which is a real cost on clips where only a medic
# speaks -- in exchange for not collapsing. Which trade is better is an empirical
# question, and that is the point of running it as a variant.
ALERTNESS_VERBAL_MINIMAL_TASK = AssessmentTask(
    name='alertness_verbal',
    classes=('normal', 'abnormal', 'absent'),
    rubric=(
        "Listen to this recording and decide which best describes the human "
        "vocalization in it.\n"
        "normal: someone speaks in calm, coherent, intelligible words -- ordinary "
        "conversational speech that makes sense.\n"
        "abnormal: the vocalization is distressed or disordered -- screaming, "
        "moaning, groaning, crying, whimpering, grunting, gasping, or speech that "
        "is slurred, mumbled, rambling or confused.\n"
        "absent: no human voice at all, only background, machine or environmental "
        "noise."
    ),
    window_caveat=(
        "You are hearing one segment of a longer recording. Judge only this segment."
    ),
    observation_module='gemma_audio_alertness_verbal',
)

# What the Gemma node and assessor run: verbal alertness only. Respiratory distress is
# assessed by AST in audio_classification_node.py, and Gemma prompting aimed at it was
# measured to damage verbal alertness (see NOTE_PROMPT).
GEMMA_TASKS: Tuple[AssessmentTask, ...] = (ALERTNESS_VERBAL_TASK,)

# The two-task tuple the 2026-09-13/14 sweeps were run with. Kept ONLY so the analysis
# scripts that read those old runs (aggregate_metrics, compare_algorithms,
# make_presentation_figures, bn_emission_model, dirichlet_calibration) still load.
# Nothing that runs Gemma uses it.
DEFAULT_TASKS: Tuple[AssessmentTask, ...] = (RESPIRATORY_DISTRESS_TASK, ALERTNESS_VERBAL_TASK)


# ---------------------------------------------------------------------------
# Symptom-level decomposition of verbal alertness.
#
# ALERTNESS_VERBAL_TASK asks the model for the clinical class directly, which
# requires it to certify the casualty is "oriented to time, person and place" --
# something it cannot check, because it does not know the correct answers to the
# responder's questions. Measured cost: 'normal' is the weakest class by a wide
# margin (recall 0.37).
#
# These two tasks ask instead for facts that ARE decidable from audio, and let a
# Bayesian network apply the rule. The rubric is a conjunction, so two binary
# questions reproduce it exactly:
#
#   vocalized=absent                        -> alertness_verbal = absent
#   vocalized=present, coherent=absent      -> alertness_verbal = abnormal
#   vocalized=present, coherent=present     -> alertness_verbal = normal
#
# The decisive practical property is that both are FULLY IDENTIFIED by the
# existing clip-level labels -- vocalized is (label != absent), coherent is
# (label == normal) among vocalizers -- so their emission parameters can be
# fitted from the ground truth this data set already has, with no new annotation.
# That is what makes the decomposition testable rather than merely plausible.
CASUALTY_VOCALIZED_TASK = AssessmentTask(
    name='casualty_vocalized',
    classes=('absent', 'present'),
    rubric=(
        "Decide whether the casualty makes any vocal sound of their own.\n"
        "present: the casualty produces any sound with their voice -- words, or a "
        "non-speech sound such as a moan, groan, cry, whimper, scream or grunt.\n"
        "absent: the casualty makes no vocal sound at all.\n"
        "Only the injured casualty's own voice counts.\n"
        "HOW TO TELL WHO IS WHO: the casualty is the person being spoken TO. They never "
        "ask questions, never give instructions, and never talk about the casualty in "
        "the third person. Any voice that asks a question ('can you hear me?', 'what is "
        "your name?'), gives a command ('squeeze my hand'), speaks in a flat synthetic "
        "robot tone, or describes the casualty to someone else is a responder or the "
        "robot -- not the casualty. A scene where responders are talking constantly and "
        "the casualty never answers is 'absent', no matter how much speech you hear."
    ),
    window_caveat=(
        "You are hearing one segment of a longer recording. Judge only this segment."
    ),
    observation_module='gemma_audio_casualty_vocalized',
    # Prompt-independent, like ALERTNESS_VERBAL_TASK's: the recordings coming next have
    # no questioner at all, so a verdict that counts prompts cannot be applied to them.
    summary_instruction=(
        "Apply the definitions to the encounter as a whole, using everything the "
        "casualty is reported to say or make a sound of in any segment. One segment is "
        "enough: a casualty heard in any segment vocalized, whatever the other segments "
        "say."),
)

CASUALTY_COHERENT_SPEECH_TASK = AssessmentTask(
    name='casualty_coherent_speech',
    classes=('absent', 'present'),
    rubric=(
        "Decide whether the casualty speaks in coherent, intelligible words.\n"
        "WHETHER ANYONE QUESTIONED THEM CHANGES THE ANSWER, so settle that first: did "
        "a robot or a responder ask the casualty anything at all?\n"
        "IF NOBODY QUESTIONED THEM -- present when the casualty produces words you can "
        "make out, whether a sentence, a shout for help, or a statement about "
        "themselves. Nothing has to be a reply. Calling out for help, even the same "
        "call many times over, is PRESENT: a person who can call for help is speaking. "
        "absent only when there are no words at all, only non-speech sound, or speech "
        "too slurred or mumbled to make out.\n"
        "IF THEY WERE QUESTIONED -- present when the replies fit what was asked or are "
        "sensible statements about themselves. absent when the replies are confused, "
        "unrelated to the question, or the same phrase over and over, such as 'I don't "
        "know, I don't know, I don't know', which is not coherent however clear each "
        "word is.\n"
        "In both cases: no words at all, or only moaning, groaning or crying, is "
        "absent.\n"
        "Only the injured casualty's own voice counts.\n"
        "HOW TO TELL WHO IS WHO: the casualty is the person being spoken TO. Voices that "
        "ask questions, give instructions, speak in a flat synthetic robot tone, or "
        "describe the casualty to someone else are responders or the robot. Their speech "
        "is fluent and coherent almost by definition and must be ignored entirely -- "
        "counting it is the single most common way to get this question wrong."
    ),
    window_caveat=(
        "You are hearing one segment of a longer recording, so a reply may answer a "
        "question asked before this segment began. Judge this segment on what is in it."
    ),
    observation_module='gemma_audio_casualty_coherent_speech',
    # WHETHER ANYONE ASKED MATTERS, and the rubric above now says so. It sits in the
    # RUBRIC rather than here because the rubric is the one piece shared by all three
    # prompts that ask this question -- the per-window one, the summary over notes, and
    # the one that reads the recogniser's transcript. Putting the rule in
    # summary_instruction alone left the transcript judge, the only one with the actual
    # words in front of it, still working from the old definition.
    #
    # What it fixes: 8 of 40 failures are recordings labelled NORMAL where coherence
    # came back 0.00 -- "Help! My leg! I need help! Anyone coming? Yes, I can!" among
    # them. The old wording said "it does not matter whether anyone asked a question",
    # so repeated shouting did not look like a conversation. It is not one. It is a
    # person who can speak.
    summary_instruction=(
        "Apply the definitions to the encounter as a whole, using every word the "
        "casualty is reported to say across all segments, and decide whether anyone "
        "questioned them using every segment rather than this one alone."),
)

# The two symptom questions a Bayesian network resolves into alertness_verbal.
SYMPTOM_TASKS: Tuple[AssessmentTask, ...] = (CASUALTY_VOCALIZED_TASK,
                                             CASUALTY_COHERENT_SPEECH_TASK)


@dataclass
class WindowAssessment:
    """
    Everything one window produced, flat enough to become one CSV row.

    :param probabilities: task name -> probability vector aligned with task.classes
    :param latency_s: task name -> seconds spent in that task's forward pass
    :param note: observation notes for this window, or None when note-taking is off
    """
    probabilities: Dict[str, np.ndarray] = field(default_factory=dict)
    latency_s: Dict[str, float] = field(default_factory=dict)
    note: Optional[str] = None

    def label(self, task: AssessmentTask) -> str:
        return task.classes[int(np.argmax(self.probabilities[task.name]))]

    def confidence(self, task: AssessmentTask) -> float:
        return float(np.max(self.probabilities[task.name]))


@dataclass
class RecordingAssessment:
    """
    The single whole-recording verdict reached from all of a recording's notes.

    :param probabilities: task name -> probability vector aligned with task.classes
    :param latency_s: task name -> seconds spent scoring that task
    :param notes: the concatenated, time-ordered notes the verdict was reached from
    """
    probabilities: Dict[str, np.ndarray] = field(default_factory=dict)
    latency_s: Dict[str, float] = field(default_factory=dict)
    notes: str = ''

    def label(self, task: AssessmentTask) -> str:
        return task.classes[int(np.argmax(self.probabilities[task.name]))]

    def confidence(self, task: AssessmentTask) -> float:
        return float(np.max(self.probabilities[task.name]))


class GemmaAudioAssessor:
    """
    Loads a unified Gemma 4 checkpoint once and scores audio windows against a
    set of AssessmentTasks.

    torch/transformers are imported inside load() rather than at module import so
    that the evaluation scripts -- which only ever read this module's task
    definitions and class orderings -- don't drag in a multi-GB CUDA stack.
    """

    def __init__(self,
                 model_id: str = 'google/gemma-4-12B-it',
                 tasks: Sequence[AssessmentTask] = GEMMA_TASKS,
                 device: str = 'cuda',
                 dtype: str = 'bfloat16',
                 logger=None,
                 note_prompt: str = NOTE_PROMPT):
        """
        :param model_id: HuggingFace id or local path of a unified (audio-capable) Gemma 4
        :param tasks: the classifications to run on every window
        :param device: torch device to place the model on
        :param dtype: torch dtype name used to load weights
        :param logger: anything with .info/.warning/.error (e.g. a rclpy logger); optional
        :param note_prompt: the prompt take_note() uses. NOTE_PROMPT is the current
                            baseline; the note-prompt test harness passes variants
        """
        self.model_id = model_id
        self.tasks = tuple(tasks)
        self.note_prompt = note_prompt
        self.device = device
        self.dtype = dtype
        self.logger = logger
        self.model = None
        self.processor = None
        # class name -> token id, shared across tasks (classes like 'absent' recur)
        self._class_token_ids: Dict[str, int] = {}
        # task name -> the prompt text, built once since it never varies per window
        self._prompts: Dict[str, str] = {}

    def _log(self, level: str, message: str) -> None:
        if self.logger is not None:
            getattr(self.logger, level)(message)

    def load(self) -> None:
        """
        Loads the checkpoint and resolves class tokens. Blocking and slow (tens of
        seconds); call once before any assess_window() call.
        :return: nothing
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        start = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=getattr(torch, self.dtype)
        ).to(self.device).eval()

        audio_capable = getattr(self.model.config, 'audio_config', None) is not None
        if not audio_capable:
            # Several Gemma 4 checkpoints (e.g. gemma-4-26b-a4b-it) ship text+vision
            # only. Failing here beats failing later with an opaque processor error.
            raise ValueError(
                f"{self.model_id} has no audio_config -- it is a text/vision checkpoint "
                f"and cannot classify audio. Use a unified checkpoint such as "
                f"google/gemma-4-12B-it or google/gemma-4-e4b-it."
            )

        self._resolve_class_tokens()
        self._prompts = {task.name: task.window_prompt() for task in self.tasks}
        self._log('info', f'Loaded {self.model_id} ({type(self.model).__name__}) '
                          f'in {time.time() - start:.1f}s')

    def _resolve_class_tokens(self) -> None:
        """
        Maps every task's class names to single token ids, asserting single-token-ness.

        The exactness of the per-window probabilities depends on this: scoring one
        logit position is only equivalent to scoring the whole answer if each class
        is one token. A checkpoint whose tokenizer splits a class word would make the
        probabilities quietly wrong, so this raises instead.
        :return: nothing
        """
        tokenizer = self.processor.tokenizer
        for task in self.tasks:
            for class_name in task.classes:
                # Leading space because the answer follows PREFILL, and because it is
                # what makes these single tokens -- see PREFILL's comment.
                token_ids = tokenizer.encode(f' {class_name}', add_special_tokens=False)
                if len(token_ids) != 1:
                    raise ValueError(
                        f"Class '{class_name}' of task '{task.name}' tokenizes to "
                        f"{len(token_ids)} tokens ({token_ids}) under {self.model_id}'s "
                        f"tokenizer; single-position logit scoring requires exactly one. "
                        f"Pick a different class word or score full sequences instead."
                    )
                self._class_token_ids[class_name] = token_ids[0]

    def _build_inputs(self, prompt: str, audio: Optional[np.ndarray] = None,
                      prefill: bool = True):
        """
        Renders the chat prompt (optionally around audio) and tokenizes it.

        The chat template is rendered to TEXT first and any waveform handed to the
        processor separately, rather than going through apply_chat_template's
        tokenize=True path: that path resolves audio content blocks by loading a
        file path or URL, which would mean writing every window to disk just to read
        it back.
        :param prompt: the fully-built question text
        :param audio: mono float32 in [-1, 1] at SAMPLE_RATE, or None for a text-only
                      prompt (the recording-level pass, which reasons over notes)
        :param prefill: append PREFILL so a scored answer lands at a known position;
                        False when generating free text
        :return: dict of model inputs already moved to self.device
        """
        content = ([{'type': 'audio'}] if audio is not None else []) + \
                  [{'type': 'text', 'text': prompt}]
        text = self.processor.apply_chat_template(
            [{'role': 'user', 'content': content}],
            add_generation_prompt=True, tokenize=False)
        if prefill:
            text += PREFILL
        kwargs = {'text': [text], 'return_tensors': 'pt'}
        if audio is not None:
            kwargs['audio'] = [audio]
        inputs = self.processor(**kwargs)
        return {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in inputs.items()}

    def _generate(self, prompt: str, audio: Optional[np.ndarray], max_new_tokens: int,
                  no_repeat_ngram_size: int = 0) -> str:
        """
        Runs greedy generation and returns just the newly generated text.

        :param prompt: the fully-built question text
        :param audio: mono float32 window, or None for a text-only prompt
        :param max_new_tokens: generation cap
        :param no_repeat_ngram_size: block verbatim repeats of this many tokens; 0 disables
        :return: the model's reply, stripped
        """
        import torch

        inputs = self._build_inputs(prompt, audio=audio, prefill=False)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                            do_sample=False,
                                            no_repeat_ngram_size=no_repeat_ngram_size)
        return self.processor.decode(generated[0][inputs['input_ids'].shape[1]:],
                                     skip_special_tokens=True).strip()

    def _score_prompt(self, prompt: str, task: AssessmentTask,
                      audio: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Scores one already-built prompt, returning a distribution over task.classes.

        :param prompt: the fully-built question, ending so that the next token is the answer
        :param task: the classification whose classes to score
        :param audio: mono float32 window, or None for a text-only prompt
        :return: float64 array aligned with task.classes, summing to 1
        """
        logits = self.score_prompt_logits(prompt, task, audio=audio)
        # Renormalizing over only the class tokens turns the full-vocabulary
        # distribution into a proper distribution over the allowed answers.
        #
        # EXPECT REPEATED PROBABILITY VALUES IN THE OUTPUT, especially for two-class
        # tasks -- this is not a bug and not a sign that the audio is being ignored.
        # Gemma 4 applies final_logit_softcapping of 30.0, so logits land in a range
        # where bf16's 8-bit mantissa has a step of ~0.125. Logit DIFFERENCES
        # therefore fall on a 0.125 lattice, and a two-class softmax of a difference
        # can only take lattice values (0.880797, 0.893309, 0.914901, 0.924142,
        # 0.932453, 0.952574, ... for differences of 2.0, 2.125, 2.375, 2.5, 2.625,
        # 3.0). Two different windows the model feels equally strongly about land on
        # the same value exactly. The resulting quantization error is ~0.015 in
        # probability, an order of magnitude below both the ECE bin width and the
        # sampling noise of a ~189-recording evaluation, so it does not meaningfully
        # move Brier or ECE; loading in float32 would halve it at twice the VRAM.
        shifted = np.exp(logits - logits.max())
        return shifted / shifted.sum()

    def score_prompt_logits(self, prompt: str, task: AssessmentTask,
                            audio: Optional[np.ndarray] = None) -> np.ndarray:
        """
        The answer-token logits behind _score_prompt(), before normalization.

        The recording-level verdict saturates -- in the 2026-09-15 summary runs 77-89%
        of recordings come out with a top probability above 0.999 -- and a distribution
        that certain, stored to six decimals, loses the ordering of the other two
        classes entirely. The logits keep it, which is what temperature calibration and
        fusion with other classifiers need.
        :param prompt: the fully-built question, ending so that the next token is the answer
        :param task: the classification whose classes to score
        :param audio: mono float32 window, or None for a text-only prompt
        :return: float64 logits aligned with task.classes
        """
        import torch

        inputs = self._build_inputs(prompt, audio=audio, prefill=True)
        with torch.inference_mode():
            # Last position's logits are the distribution over the token that would
            # follow PREFILL, i.e. over the answer word itself.
            logits = self.model(**inputs).logits[0, -1].float()
        class_ids = torch.tensor([self._class_token_ids[c] for c in task.classes],
                                 device=logits.device)
        return logits[class_ids].cpu().numpy().astype(np.float64)

    def score_task(self, audio: np.ndarray, task: AssessmentTask,
                   note: Optional[str] = None,
                   robot_speech: Optional[Sequence[str]] = None) -> np.ndarray:
        """
        Returns the probability vector over task.classes for one audio window.

        :param audio: mono float32 in [-1, 1] at SAMPLE_RATE, at most MAX_AUDIO_SECONDS
        :param task: the classification to perform
        :param note: this window's observation notes, if scoring should be grounded
        :param robot_speech: verbatim robot utterances during this window
        :return: float64 array aligned with task.classes, summing to 1
        """
        # The cached prompt covers the common no-context case; anything grounded is
        # window-specific and has to be rebuilt.
        prompt = (self._prompts[task.name] if not (note or robot_speech)
                  else task.window_prompt(note=note, robot_speech=robot_speech))
        return self._score_prompt(prompt, task, audio=audio)

    def take_note(self, audio: np.ndarray, max_new_tokens: int = 200,
                  note_prompt: Optional[str] = None) -> str:
        """
        Writes structured observation notes for one window, using self.note_prompt --
        see NOTE_PROMPT.

        :param audio: mono float32 in [-1, 1] at SAMPLE_RATE
        :param max_new_tokens: generation cap. A few short headings need well under
                               this; the cap's real job is bounding a window where
                               generation goes wrong, since anything produced here is
                               fed verbatim to assess_recording()
        :param note_prompt: overrides self.note_prompt for this window only, which is
                            what lets the caller append per-window evidence (such as a
                            sound detector's findings) to an otherwise fixed prompt
        :return: the notes as text
        """
        return self._generate(note_prompt or self.note_prompt, audio, max_new_tokens,
                              no_repeat_ngram_size=NOTE_NO_REPEAT_NGRAM)

    def transcribe(self, audio: np.ndarray, max_new_tokens: int = 128) -> str:
        """
        Transcribes speech in the window. Not used by the evaluation path, which uses
        take_note() instead; kept here so the Whisper replacement lives alongside the
        classifier it replaces, for the live robot.

        :param audio: mono float32 in [-1, 1] at SAMPLE_RATE
        :param max_new_tokens: generation cap
        :return: transcribed text, empty string if nothing intelligible was said
        """
        prompt = (f"{_CONTEXT}\n\nTranscribe any intelligible speech in this recording "
                  f"verbatim. If nobody says anything intelligible, reply with exactly: "
                  f"(no speech)")
        reply = self._generate(prompt, audio, max_new_tokens)
        return '' if reply == '(no speech)' else reply

    def assess_recording(self, notes: Sequence[Tuple[float, float, str]]
                         ) -> RecordingAssessment:
        """
        Reaches one verdict per task for a whole recording from its window notes.

        This is the half of the pipeline that can actually apply a rubric phrased in
        terms of attempts -- "responsive after at most 2 attempts", "no vocalization
        after 2 speech prompts". Those count events across the entire encounter, so
        no single 30 s window has the evidence to decide them, and no way of pooling
        per-window probabilities can reconstruct it: a window holding an unanswered
        prompt and a window holding the answer both look ambiguous alone, while the
        pair is unambiguous. Pooling averages that ambiguity; this reads the sequence.

        Scoring still comes from logits over the class tokens, exactly as the
        per-window path does, so both decision paths produce comparable
        probabilities for Brier/ECE rather than one being a parsed word.
        :param notes: (start_s, end_s, note) per window, in time order
        :return: the whole-recording probabilities, latencies, and the notes used
        """
        if self.model is None:
            raise RuntimeError('assess_recording() called before load()')
        rendered = '\n\n'.join(
            f'[segment {index + 1}: {start:.0f}-{end:.0f}s]\n{note}'
            for index, (start, end, note) in enumerate(notes))

        result = RecordingAssessment(notes=rendered)
        for task in self.tasks:
            start = time.time()
            result.probabilities[task.name] = self._score_prompt(
                task.summary_prompt(rendered), task)
            result.latency_s[task.name] = time.time() - start
        return result

    def assess_window(self, audio: np.ndarray, take_notes: bool = False,
                      ground_in_note: bool = False,
                      robot_speech: Optional[Sequence[str]] = None,
                      note_prompt: Optional[str] = None) -> WindowAssessment:
        """
        Runs every configured task (and optionally note-taking) on one window.

        The per-window scores are kept even when note-taking is on, for two reasons:
        they cost ~0.3 s next to note generation's several seconds, and keeping both
        is what lets the evaluation compare window-pooling against the
        notes-then-verdict path on identical audio instead of arguing about it.
        :param audio: mono float32 in [-1, 1] at SAMPLE_RATE
        :param take_notes: also generate observation notes for assess_recording()
        :param ground_in_note: score the tasks conditioned on this window's note
                               rather than on the audio alone. Requires take_notes,
                               and reorders the work: the note is written first and
                               then fed into every scored question.
        :param robot_speech: verbatim robot utterances overlapping this window
        :param note_prompt: per-window override of the note prompt -- see take_note()
        :return: the window's probabilities, per-task latencies, and notes
        """
        if ground_in_note and not take_notes:
            raise ValueError('ground_in_note requires take_notes')
        if self.model is None:
            raise RuntimeError('assess_window() called before load()')
        max_samples = int(MAX_AUDIO_SECONDS * SAMPLE_RATE)
        if len(audio) > max_samples:
            # The processor would truncate silently; be loud and deterministic instead.
            self._log('warning', f'Window of {len(audio) / SAMPLE_RATE:.1f}s exceeds the '
                                 f'{MAX_AUDIO_SECONDS:.0f}s model limit; using the last '
                                 f'{MAX_AUDIO_SECONDS:.0f}s.')
            audio = audio[-max_samples:]

        result = WindowAssessment()
        # Notes first when the scoring is grounded in them -- the whole point is that
        # the model commits to an attribution before being asked to judge.
        if take_notes:
            start = time.time()
            result.note = self.take_note(audio, note_prompt=note_prompt)
            result.latency_s['note'] = time.time() - start
        for task in self.tasks:
            start = time.time()
            result.probabilities[task.name] = self.score_task(
                audio, task,
                note=result.note if ground_in_note else None,
                robot_speech=robot_speech)
            result.latency_s[task.name] = time.time() - start
        return result


def pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    """
    Converts int16 PCM to the float32 range the audio feature extractor expects.
    :param pcm: int16 samples
    :return: float32 samples in [-1, 1]
    """
    return pcm.astype(np.float32) / 32768.0


def pool_probabilities(window_probs: Sequence[np.ndarray], method: str = 'mean') -> np.ndarray:
    """
    Reduces a recording's per-window probability vectors to one recording-level
    vector, since the validation set labels whole clips but the node classifies
    windows.

    The choice matters for calibration, not just accuracy, so it is explicit and
    swappable rather than hardcoded:
      mean      -- average the distributions. Well-behaved and keeps confidences
                   honest; a label only audible in part of the clip gets diluted.
      max_conf  -- take the single most confident window's distribution. Suits
                   "happened at some point" labels but is systematically
                   over-confident, which inflates ECE.
      logit_mean-- average in log space then renormalize (geometric mean). Sits
                   between the two; sharper than mean, less extreme than max_conf.
      median    -- element-wise median, renormalized. Robust to a few wild windows.

    :param window_probs: per-window probability vectors, all the same length
    :param method: one of 'mean', 'max_conf', 'logit_mean', 'median'
    :return: one probability vector summing to 1
    """
    stacked = np.asarray(window_probs, dtype=np.float64)
    if stacked.ndim != 2 or stacked.shape[0] == 0:
        raise ValueError(f'Expected a non-empty 2-D array of probabilities, got {stacked.shape}')

    if method == 'mean':
        pooled = stacked.mean(axis=0)
    elif method == 'max_conf':
        pooled = stacked[int(np.argmax(stacked.max(axis=1)))]
    elif method == 'logit_mean':
        pooled = np.exp(np.log(np.clip(stacked, 1e-12, 1.0)).mean(axis=0))
    elif method == 'median':
        pooled = np.median(stacked, axis=0)
    else:
        raise ValueError(f"Unknown pooling method '{method}'")
    return pooled / pooled.sum()


POOLING_METHODS: Tuple[str, ...] = ('mean', 'max_conf', 'logit_mean', 'median')
