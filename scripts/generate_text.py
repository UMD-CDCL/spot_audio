import torch
from TTS.api import TTS


PROMPT = "Do you need help?"
OUTPUT_PATH = "v1/do_you_need_help.wav"

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS(model_name="tts_models/en/ek1/tacotron2", progress_bar=False).to(device)

# Text to speech list of amplitude values as output
tts.tts_to_file(text=PROMPT, file_path=OUTPUT_PATH)


