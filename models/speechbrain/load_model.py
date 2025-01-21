import torch, torchaudio
from functools import partial
import argparse
import os
from transformers import MCTCTForCTC, MCTCTProcessor
from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.inference.ASR import EncoderASR

class_dict = {
    'speechbrain/asr-crdnn-rnnlm-librispeech': EncoderDecoderASR,
    'speechbrain/asr-wav2vec2-switchboard': EncoderASR,
    'speechbrain/asr-branchformer-large-tedlium2': EncoderDecoderASR,
    'speechbrain/asr-transformer-switchboard': EncoderDecoderASR,
}

@torch.inference_mode()
def pipeline(model, device, waveform):
    assert isinstance(waveform, str), f'waveform must be a file path, but got {type(waveform)}'
    assert os.path.exists(waveform), f'file {waveform} does not exist'

    transcript = model.transcribe_file(waveform)

    return transcript

def load(args):
    device = args.device if args.device != '' else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = class_dict[args.name].from_hparams(source=args.name, savedir=args.save_dir, run_opts={'device': device})
    device = torch.device(device)
    model = model.to(device)
    return partial(pipeline, model, device)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='speechbrain/asr-branchformer-large-tedlium2')
    parser.add_argument('--device', type=str, default='', choices=['', 'cpu', 'cuda'])
    parser.add_argument('--save_dir', type=str, default='')
    args, _ = parser.parse_known_args()
    args.save_dir = None if args.save_dir == '' else args.save_dir
    return args
