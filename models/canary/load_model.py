from nemo.collections.asr.models import EncDecMultiTaskModel
import torch
from functools import partial
import argparse
import torchaudio
import os

@torch.inference_mode()
def pipeline(model, waveform):
    assert isinstance(waveform, str), 'only implemented for file paths for now'
    assert os.path.exists(waveform), f'file {waveform} does not exist'
    predicted_text = model.transcribe([waveform])[0]
    return predicted_text

def load(args):
    canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
    return partial(pipeline, canary_model)

def get_args():
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    return args
