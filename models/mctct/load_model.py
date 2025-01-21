import torch, torchaudio
from functools import partial
import argparse
import os
from transformers import MCTCTForCTC, MCTCTProcessor

@torch.inference_mode()
def pipeline(model, processor, device, waveform):
    sr = 16000
    if isinstance(waveform, str):
        assert os.path.exists(waveform), f'file {waveform} does not exist'
        waveform, sr = torchaudio.load(waveform)
        if sr != 16000: waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.dim() != 1: waveform = waveform[0]
    elif isinstance(waveform, torch.Tensor):
        if waveform.dim() != 1: waveform = waveform[0]
    else:
        raise ValueError(f'waveform must be a file path or a torch.Tensor, but got {type(waveform)}')
    
    input_features = processor(waveform, sampling_rate=16000, return_tensors="pt").input_features 
    input_features = input_features.to(device)
    logits = model(input_features).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    #print('\n---',transcription[0],'---\n')
    return transcription[0]

def load(args):
    model = MCTCTForCTC.from_pretrained("speechbrain/m-ctc-t-large")
    processor = MCTCTProcessor.from_pretrained("speechbrain/m-ctc-t-large")
    device = args.device if args.device != '' else 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model = model.to(device)

    return partial(pipeline, model, processor, device)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='', choices=['', 'cpu', 'cuda'])
    args, _ = parser.parse_known_args()
    return args
