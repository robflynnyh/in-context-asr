import torch, torchaudio
from functools import partial
import argparse
import os
from transformers import MoonshineForConditionalGeneration, AutoProcessor

class_dict = {
    'tiny': 'UsefulSensors/moonshine-tiny',
    'base': 'UsefulSensors/moonshine-base'
}

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
    
    input_features = processor(waveform, sampling_rate=16000, return_tensors="pt")
    input_features = input_features.to(device)
    token_limit_factor = 6.5 / processor.feature_extractor.sampling_rate
    seq_lens = input_features.attention_mask.sum(dim=-1)
    max_length = int((seq_lens * token_limit_factor).max().item())
    
    generated_ids = model.generate(**input_features, max_length=max_length)

    transcription = processor.decode(generated_ids[0], skip_special_tokens=True)

    print('-----------', transcription, '-----------')
    return transcription

def load(args):
    model = MoonshineForConditionalGeneration.from_pretrained(class_dict[args.name])
    processor = AutoProcessor.from_pretrained(class_dict[args.name])

    device = args.device if args.device != '' else 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model = model.to(device)

    return partial(pipeline, model, processor, device)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='', choices=['', 'cpu', 'cuda'])
    parser.add_argument('--name', type=str, default='tiny', choices=['tiny', 'base'])
    args, _ = parser.parse_known_args()
    return args
