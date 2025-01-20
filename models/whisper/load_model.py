import whisper
import argparse
from functools import partial
import torch
import os 

@torch.inference_mode()
def pipeline(model, waveform):
    assert isinstance(waveform, str), 'only implemented for file paths for now'
    assert os.path.exists(waveform), f'file {waveform} does not exist'
    return model.transcribe(waveform)['text']
   
def load(args):
    model = whisper.load_model(args.model_name)
    return partial(pipeline, model)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    args, _ = parser.parse_known_args()
    return args
