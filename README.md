# Overview
`spot_audio` is a ROS2 package consisting of two nodes:
- `spot_microphone_node.py`
- `spot_speaker_node.py`


### Microphone and Audio Processing
Spot uses the [ReSpeaker Microphone Array v2.0](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/) (or "respeaker" for short) to collect audio data. The respeaker consists of four microphones in a circular array, but we only use one microphone.

Spot makes use of two deep-learning models to perform its listening tasks: [Whisper](https://github.com/openai/whisper) and [Audio Spectrogram Transformer](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer) (AST).

Whisper is an off-the-shelf speech-to-text (stt) model Spot uses to transcribe any speech it hears. We use the [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a `python3` wrapper for this model.

[AST](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer) is an off-the-shelf audio classifier Spot uses to detect: speech, nonverbal vocalizations, and respiratory distress. We use a `python3` package to interact with this model.


The `spot_microphone_node.py` is the `ros2` node that processes the raw audio from mic. The node publishes four `ros2` topics and serves a single `ros2` service:
1. `String.msg` published on `/<SPOT_NAME>/heart_beat` topic. We publish to this topic every 8 seconds to alert `spot_state_manager` node that the microphone node is live.
2. `Speech.msg` published on `/<SPOT_NAME/speech` topic. We publish to this topic to every 3 seconds if `whisper` picked up on any speech.
3. `AudioData.msg` published on `/<SPOT_NAME/raw_audio` topic. We publish to this topic continuously as we collect data from the microphone.
4. `Observation.msg` published on `/<SPOT_NAME/observations_no_id` topic. This gets published whenever `whisper` picks up on speech or the `AST` detects speech, a non-verbal vocalization, or respiratory distress.
5. `StopListening.srv` served on `/<SPOT_NAME/stop_listening_service_name`. The `spot_speaker_node.py` calls this service whenever it's about to play audio containing speech, so that Spot doesn't accidentally transcribe its own speech.


The `spot_microphone_node.py` uses two threads whose jobs are:
1. To poll the microphone for raw audio data and put it onto a buffer, and
2. To process the audio buffer using `whisper` and `AST`.

The second thread processes the audio...


#### Outstanding Problems
The noise produced by Spot walking around drowns out all other audio data in the microphone. This limits us to using the microphone when Spot isn't moving. To address this, we plan on purchasing a directional microphone, which will hopefully attenuate all noise not coming in the direction of the speaker. The integration of this microphone into this software stack may take substantial time.

Whisper's transcriptions are not perfect and tend not to pick up on the casualty's speech. Whisper's transcriptions are better when the audio segment is longer. We restrict ourselves to processing audio segments of three seconds, because if we wait longer, then Spot's responses seem very delayed compared to when the speaker finishes talking. To resolve this problem, some minor refinements should be made to how the audio data is processed within the node.

Rather than publishing a custom `heart_beat` message, we should be using the [`diagnostic_message`](https://github.com/ros/diagnostics) ROS2 package.

### Speaker



### Style-Guide
`import` statements should be in alphabetical order.

Lists of dependencies in `package.xml` or `CMakeLists.txt` should be in alphabetical order.