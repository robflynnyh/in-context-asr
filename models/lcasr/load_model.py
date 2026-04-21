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

def _make_spec(waveform):
    if isinstance(waveform, str):
        return processing_chain(waveform, normalise=True)
    return to_spectogram(waveform, global_normalisation=True)


@torch.inference_mode()
def predict(model, waveform):
    spec = _make_spec(waveform)
    logits = model(spec.to(model.device))
    return logits


def pipeline(model, decoder, waveform):
    text = decoder(predict(model, waveform)['final_posteriors'])
    return text




def self_train_pipeline(model, tokenizer, decoder, checkpoint_config, st_args, waveform):
    """Per-utterance self-training adaptation then greedy decode."""
    # import lazily so `--model lcasr` without --self_train doesn't require dynamic_eval deps
    from models.lcasr.dynamic_eval import dynamic_eval_ctc_loss

    spec = _make_spec(waveform)
    seq_len = checkpoint_config.get('sequence_scheduler', {}).get(
        'max_sequence_length',
        checkpoint_config.get('audio_chunking', {}).get('size', 16384),
    )
    overlap = round(seq_len * 0.875)

    args = lcasr.utils.general.argsclass(
        config=checkpoint_config,
        optim_lr=st_args.optim_lr,
        spec_augment_n_time_masks=st_args.spec_augment_n_time_masks,
        spec_augment_zero_masking=st_args.spec_augment_zero_masking,
        spec_augment_n_freq_masks=st_args.spec_augment_n_freq_masks,
        spec_augment_freq_mask_param=st_args.spec_augment_freq_mask_param,
        epochs=st_args.epochs,
        shuffle=st_args.shuffle,
        lm_tta_beams=0,
    )

    logits = dynamic_eval_ctc_loss(
        args,
        model,
        spec.to(model.device),
        seq_len=seq_len,
        overlap=overlap,
        tokenizer=tokenizer,
        use_tqdm=False,
        beam_search_fn=None,
        verbose=st_args.verbose,
    )
    return decoder(torch.as_tensor(logits))


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

    decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=model.decoder.num_classes - 1)
    decode_fn = partial(decode, decoder)

    if getattr(args, 'self_train', False):
        pipeline_fn = partial(self_train_pipeline, model, tokenizer, decoder, model_config, args)
    else:
        pipeline_fn = partial(pipeline, model, decode_fn)

    return pipeline_fn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--name', type=str, default='SCConformerXL')
    # Self-training / dynamic-eval flags
    parser.add_argument('--self_train', action='store_true',
                        help='Adapt the model per utterance via pseudo-label CTC self-training before transcribing.')
    parser.add_argument('--optim_lr', type=float, default=9e-5)
    parser.add_argument('--epochs', type=int, default=1,
                        help='Self-training adaptation epochs per utterance (paper uses 5).')
    parser.add_argument('--shuffle', action='store_true', default=True)
    parser.add_argument('--spec_augment_n_time_masks', type=int, default=0)
    parser.add_argument('--spec_augment_n_freq_masks', type=int, default=6)
    parser.add_argument('--spec_augment_freq_mask_param', type=int, default=34)
    parser.add_argument('--spec_augment_zero_masking', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args, _ = parser.parse_known_args()
    return args
