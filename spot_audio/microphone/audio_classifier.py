from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import os
import torch
from transformers import ASTConfig, AutoFeatureExtractor, ASTForAudioClassification
from typing import Optional, Tuple, Dict


def pcm16_to_tensor(data_i16: npt.NDArray) -> torch.Tensor:
    """
    output a tensor whose sample mean is 0 and sample standard deviation is 0.5 as per recommendation in link below

    https://huggingface.co/docs/transformers/main/en/model_doc/audio-spectrogram-transformer#transformers.ASTForAudioClassification.forward.example
    :param data_i16: pcm16 vector coming from microphone or wave file
    :return: a normalized torch.Tensor representation of the sound
    """
    tensor = torch.from_numpy(data_i16.astype(np.float64))
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    return (tensor - mean) / (2 * std)


def audio_set_label_to_respiratory_distress_label(audio_set_label_id: int) -> Optional[int]:
    if audio_set_label_id == 22 or audio_set_label_id == 24 or audio_set_label_id == 25 or 38 <= audio_set_label_id <= 50:
        return 1  # resp. distress present
    return None


def audio_set_label_to_verbal_alertness_label(audio_set_label_id: int) -> Optional[int]:
    if 0 <= audio_set_label_id <= 7 or 27 <= audio_set_label_id <= 36 or audio_set_label_id == 68:
        return 0  # speech
    if 8 <= audio_set_label_id <= 26 or 37 <= audio_set_label_id <= 49:
        return 1  # non-verbal vocalization
    return None


class AudioClassificationStrategy(ABC):
    def __init__(self, path_to_classifier: str, logger):
        self.path_to_classifier = path_to_classifier
        self.path_to_preprocessor = os.path.join(path_to_classifier, 'preprocessor_config.json')
        self.classifier_feature_extractor = None
        self.classifier_model = None
        self.device = 'cuda'
        self.logger = logger

    def classify_audio(self, audio_data: npt.NDArray) -> Optional[torch.Tensor]:
        if self.classifier_feature_extractor is not None:
            # convert the PCM16 raw audio vector to a pytorch tensor conforming to requirements of input of model
            audio_tensor = pcm16_to_tensor(audio_data)

            # preprocess the audio
            inputs = self.classifier_feature_extractor(audio_tensor, sampling_rate=16000.0, return_tensors="pt")
            tensor = inputs['input_values'].to(self.device)
            return tensor
        return None

    def load_classifier(self) -> bool:
        if os.path.isdir(self.path_to_classifier) and os.path.isfile(self.path_to_preprocessor):
            self.classifier_feature_extractor = AutoFeatureExtractor.from_pretrained(self.path_to_preprocessor)
            self.classifier_model = ASTForAudioClassification.from_pretrained(self.path_to_classifier)
            self.classifier_model.to(self.device)
            return True
        return False

    @abstractmethod
    def apply_strategy(self, output_tensor: Optional[torch.Tensor]) -> Tuple[int, int]:
        pass


class MaxArgStrategy(AudioClassificationStrategy):
    def __init__(self, path_to_classifier: str, logger):
        super().__init__(path_to_classifier, logger)

    def apply_strategy(self, output_tensor: Optional[torch.Tensor]) -> dict[str, int | None] | None:
        if self.classifier_model is not None and output_tensor is not None:
            # use the model to predict the audio set class
            with torch.no_grad():
                logits = self.classifier_model(output_tensor).logits

            # determine the max label
            predicted_audio_set_label_id = torch.argmax(logits, dim=-1).item()

            # determine the name of the label the model predicted
            predicted_audio_set_label = self.classifier_model.config.id2label[predicted_audio_set_label_id]

            # convert the AudioSet label to a casualty report label
            predicted_verbal_alertness_label = audio_set_label_to_verbal_alertness_label(predicted_audio_set_label_id)
            predicted_respiratory_distress_label = audio_set_label_to_respiratory_distress_label(predicted_audio_set_label_id)

            self.logger.info(f"Predicted AudioSet label: {predicted_audio_set_label}")

            # return the casualty report id
            return {"alertness_verbal": predicted_verbal_alertness_label, "respiratory_distress": predicted_respiratory_distress_label}
        return None


class NNStrategy(AudioClassificationStrategy):
    def apply_strategy(self, output_tensor: Optional[torch.Tensor]) -> dict[str, int | None] | None:
        """

        INPUT

        :param audio_data: PCM 16 data
        :return:
        """
        pass

