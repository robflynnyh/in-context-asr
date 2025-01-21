import nemo.collections.asr as nemo_asr
import torch
from functools import partial
import argparse
import os

@torch.inference_mode()
def pipeline(model, waveform):
    assert isinstance(waveform, str), 'only implemented for file paths for now'
    assert os.path.exists(waveform), f'file {waveform} does not exist'
    predicted_text = model.transcribe([waveform])
    while isinstance(predicted_text, list) or isinstance(predicted_text, tuple): predicted_text = predicted_text[0]
    return predicted_text

def load(args):
    if args.type == 'ctc':
        name = "nvidia/parakeet-ctc-0.6b"
        if args.name != '': name = args.name
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=name)
    elif args.type == 'rnnt':
        name = "nvidia/parakeet-rnnt-0.6b"
        if args.name != '': name = args.name
        asr_model = nemo_asr.models.EncDecRNNTModel.from_pretrained(model_name=name)

    return partial(pipeline, asr_model)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['ctc', 'rnnt'], default='ctc')
    parser.add_argument('--name', type=str, default='')
    args, _ = parser.parse_known_args()
    return args
