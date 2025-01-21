from lcasr.utils.general import load_model, get_model_class
from lcasr.utils.audio_tools import to_spectogram, processing_chain
import lcasr
import torch
from lcasr.decoding.greedy import GreedyCTCDecoder
from functools import partial
import argparse
import torchaudio

def decode(decoder, logits):
    return decoder(torch.as_tensor(logits).squeeze(0))

@torch.inference_mode()
def predict(model, waveform):
    if isinstance(waveform, str): spec = processing_chain(waveform, normalise=True)
    else: spec = to_spectogram(waveform, global_normalisation=True)
    logits = model(spec.to(model.device))
    return logits

def pipeline(model, decoder, waveform):
    return decoder(predict(model, waveform)['final_posteriors'])

def load(args):
    checkpoint, model_class = args.checkpoint, args.name

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    checkpoint = torch.load(checkpoint, map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_config = checkpoint['config']
    model = load_model(model_config, tokenizer.vocab_size(), model_class=get_model_class({'model_class': model_config.get('model_class', model_class)}))
    model.load_state_dict(checkpoint['model'], strict=False)
    model.device = device
    model = model.to(device)

    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)
    decode_fn = partial(decode, decoder)
    pipeline_fn = partial(pipeline, model, decode_fn)

    return pipeline_fn

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--name', type=str, default='SCConformerXL')
    args, _ = parser.parse_known_args()
    return args
