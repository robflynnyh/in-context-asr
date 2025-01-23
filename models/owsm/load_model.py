from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
from espnet2.bin.s2t_inference import Speech2Text
import torch
import argparse
from functools import partial
import os
import librosa

@torch.inference_mode()
def pipeline(
        model,
        waveform
    ):
    assert isinstance(waveform, str), 'only implemented for file paths for now'
    assert os.path.exists(waveform), f'file {waveform} does not exist'
    speech, rate = librosa.load(waveform, sr=16000)
    max_length = 30 * 16000
    assert speech.shape[-1] <= max_length, f'input waveform too long, max length is {speech.shape} samples'
    speech = librosa.util.fix_length(speech, size=(16000 * 30))
    predicted_text = model(speech)[0][-2]

    #print('\n',predicted_text,'-------------------\n')
    return predicted_text

def load(args):
    device = args.device if args.device != '' else 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.type == 'ctc':
        s2t = Speech2TextGreedySearch.from_pretrained(
            "espnet/owsm_ctc_v3.1_1B" if args.name == '' else args.name,
            device=device,
            use_flash_attn=False,   # set to True for better efficiency if flash attn is installed and dtype is float16 or bfloat16
            lang_sym=args.lang_sym,
            task_sym=args.task_sym,
        )
        return partial(pipeline, s2t)
    
    elif args.type == 'encdec':
        s2t = Speech2Text.from_pretrained(
            model_tag="espnet/owsm_v3.1_ebf" if args.name == '' else args.name,
            device="cuda",
            beam_size=args.beam_size,
            ctc_weight=0.0,
            maxlenratio=0.0,
            # below are default values which can be overwritten in __call__
            lang_sym=args.lang_sym,
            task_sym=args.task_sym,
            predict_time=False,
        )
        return partial(pipeline, s2t)
    else:
        raise NotImplementedError(f'args.type {args.type} not implemented')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='ctc', choices=['ctc', 'encdec'])
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--lang_sym', type=str, default='<eng>') # eng for v3+ en for v1-v2!
    parser.add_argument('--task_sym', type=str, default='<asr>')
    parser.add_argument('--beam_size', type=int, default=5)
    args, _ = parser.parse_known_args()
    return args
