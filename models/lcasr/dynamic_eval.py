"""
Self-training / dynamic evaluation for LCASR.

Copied from https://github.com/robflynnyh/Self-Train-Before-You-Transcribe
(main.py::dynamic_eval_ctc_loss) so the in-context-asr repo does not need to
pull in the whole self-train repo to use it.

The function adapts the model per-utterance with a pseudo-label CTC loss
against augmented copies of the input spectrogram, then restores the original
parameters before returning so subsequent calls start from the same baseline.
"""

import random
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from lcasr.utils.augmentation import SpecAugment
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.optim import madgrad


def _replace_with_frame(spec):
    for i, _ in enumerate(spec):
        random_index = random.randint(0, spec.shape[-1])
        spec[i] = spec[i, :, :] * 0 + spec[i, :, random_index, None]
    return spec


def frame_shuffle(spec, time_dimension=False, freq_dimension=False):
    if time_dimension:
        spec = spec[:, :, torch.randperm(spec.shape[-1])]
    if freq_dimension:
        spec = spec[:, torch.randperm(spec.shape[-2]), :]
    return spec


def add_random_noise(spec, noise_factor):
    if noise_factor == 0:
        return spec
    noise = torch.normal(0, std=spec.std(), size=spec.shape).to(spec.device)
    return spec + noise * noise_factor


def cutout(spec, seq_len, cutout_val='mean', num_rectangles=5, max_width=100, max_height=10):
    if num_rectangles == 0:
        return spec

    spec_n = spec.shape[-1]
    ratio = spec_n / seq_len
    num_rectangles = int(num_rectangles * ratio)
    if num_rectangles == 0:
        return spec

    widths = torch.randint(1, max_width, (num_rectangles,))
    heights = torch.randint(1, max_height, (num_rectangles,))
    start_positions_x = torch.randint(0, spec.shape[-1], (num_rectangles,))
    end_positions_x = (start_positions_x + widths).clamp(max=spec.shape[-1])
    start_positions_y = torch.randint(0, spec.shape[-2], (num_rectangles,))
    end_positions_y = (start_positions_y + heights).clamp(max=spec.shape[-2])

    if cutout_val == 'mean_recording':
        mask_value = spec.mean()
    elif cutout_val == 'mean':
        mask_values = []
        for i in range(num_rectangles):
            mask_values.append(
                spec[:, start_positions_y[i]:end_positions_y[i],
                     start_positions_x[i]:end_positions_x[i]].mean()
            )

    for i in range(num_rectangles):
        if cutout_val == 'mean':
            spec[:, start_positions_y[i]:end_positions_y[i],
                 start_positions_x[i]:end_positions_x[i]] = mask_values[i]
        elif cutout_val == 'mean_recording':
            spec[:, start_positions_y[i]:end_positions_y[i],
                 start_positions_x[i]:end_positions_x[i]] = mask_value
        elif cutout_val == 'zero':
            spec[:, start_positions_y[i]:end_positions_y[i],
                 start_positions_x[i]:end_positions_x[i]].zero_()
    return spec


def get_specaugment_config_from_args(args):
    sa = {k.replace('spec_augment_', ''): v for k, v in args.__dict__.items() if k.startswith('spec_augment')}
    return {
        'n_time_masks': sa.get('n_time_masks', 0),
        'n_freq_masks': sa.get('n_freq_masks', 0),
        'freq_mask_param': sa.get('freq_mask_param', 42),
        'time_mask_param': sa.get('time_mask_param', -1),
        'min_p': sa.get('min_p', 0.05),
        'zero_masking': sa.get('zero_masking', False),
    }


def get_frame_shuffle_config_from_args(args):
    fs = {k.replace('frame_shuffle_', ''): v for k, v in args.__dict__.items() if k.startswith('frame_shuffle_')}
    return {
        'time_dimension': fs.get('time_dimension', False),
        'freq_dimension': fs.get('freq_dimension', False),
    }


def get_lr_args_from_args(args):
    lr = {k.replace('optim_', ''): v for k, v in args.__dict__.items() if k.startswith('optim_')}
    lr['lr'] = lr.get('lr', 9e-5)
    return lr


def get_cutout_params_from_args(args, seq_len):
    co = {k.replace('cutout_', ''): v for k, v in args.__dict__.items() if k.startswith('cutout_')}
    return {
        'seq_len': seq_len,
        'cutout_val': co.get('value', 'mean'),
        'num_rectangles': co.get('num_rectangles', 0),
        'max_width': co.get('max_width', 100),
        'max_height': co.get('max_height', 10),
    }


def prepare_chunks(spec, seq_len, overlap):
    spec_n = spec.shape[-1]
    last_ulen, kill_next = None, False

    if spec_n <= seq_len:
        return {0: spec}, [0]

    training_data = {}
    for i in range(0, spec_n, seq_len - overlap):
        audio_chunk = spec[:, :, i:i + seq_len]
        u_len = audio_chunk.shape[-1]
        if kill_next:
            break
        elif last_ulen is not None and u_len < last_ulen:
            kill_next = True
        last_ulen = u_len
        training_data[i] = audio_chunk
    return training_data, list(training_data.keys())


def dynamic_eval_ctc_loss(
    args,
    model: nn.Module,
    spec: torch.Tensor,
    seq_len: int,
    overlap: int,
    tokenizer,
    use_tqdm: bool = True,
    optim: optim.Optimizer = madgrad.MADGRAD,
    optimizer_state: dict = None,
    beam_search_fn: Callable = None,
    return_params: bool = False,
    verbose: bool = False,
):
    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']

    spec_augment_config = get_specaugment_config_from_args(args)
    random_noise = args.__dict__.get('random_noise', 0.0)

    lr_args = get_lr_args_from_args(args)
    frame_shuffle_args = get_frame_shuffle_config_from_args(args)

    cutout_args = get_cutout_params_from_args(args, seq_len)
    if verbose:
        print(spec_augment_config, lr_args, frame_shuffle_args, cutout_args)
    num_negatives = 1

    original_model_params = [p.clone().detach().cpu() for p in model.parameters()]

    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes - 1, reduction='sum')

    optimizer = optim(model.parameters(), **lr_args)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=model.decoder.num_classes - 1)
    augmentation = SpecAugment(**spec_augment_config)

    if seq_len > spec_n:
        seq_len, overlap = spec_n, 0
    else:
        overlap = overlap if overlap != -1 else args.config['audio_chunking']['overlap']

    assert args.config['training'].get("max_seq_len", 0) == 0, 'caching is not used anymore'
    assert overlap / downsampling_factor == overlap // downsampling_factor, \
        'Overlap must be a multiple of the downsampling factor'
    if verbose:
        print(f'Using seq_len: {seq_len} and overlap: {overlap}')

    all_logits = torch.zeros((1, spec_n // 4 + seq_len, tokenizer.vocab_size() + 1))
    logit_count = torch.zeros((1, spec_n // 4 + seq_len, tokenizer.vocab_size() + 1))

    epochs = args.__dict__.get('epochs', 1)
    shuffle = args.__dict__.get('shuffle', False)
    online = args.__dict__.get('online', False)
    beams = args.__dict__.get('lm_tta_beams', 0)
    epochs = 1 if online else epochs
    shuffle = False if online else shuffle
    model_outputs = {}

    print_runtimes = args.__dict__.get('print_runtimes', False)
    if print_runtimes:
        print('Spectrogram length:', spec_n)

    model.eval()  # don't update batchrenorm stats
    training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
    for epoch in range(epochs):
        if verbose:
            print(f'Epoch {epoch + 1} / {epochs}')
        training_keys = list(training_data.keys())
        training_keys = random.sample(training_keys, len(training_keys)) if shuffle else training_keys

        epochs_stime = time.time()
        pbar = tqdm(training_keys) if use_tqdm else training_keys
        for i in pbar:
            audio_chunk = training_data[i].clone()
            audio_chunk = audio_chunk.repeat(num_negatives + 1, 1, 1)
            audio_chunk[:num_negatives] = augmentation(audio_chunk[:num_negatives])
            audio_chunk[:num_negatives] = frame_shuffle(audio_chunk[:num_negatives], **frame_shuffle_args)
            audio_chunk[:num_negatives] = add_random_noise(audio_chunk[:num_negatives], noise_factor=random_noise)
            audio_chunk[:num_negatives] = cutout(audio_chunk[:num_negatives], **cutout_args)

            u_len = audio_chunk.shape[-1]
            audio_chunk = audio_chunk.to(model.device)
            out = model(audio_signal=audio_chunk)

            if beam_search_fn is None or beams == 0:
                pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu())
            else:
                bs = beam_search_fn(log_probs=out['final_posteriors'][-1].detach().cpu(), beam_width=beams)
                bs.run_search(use_tqdm=True)
                pseudo_targets = bs.return_text(idx=0)

            if verbose:
                noisy_predictions = decoder(out['final_posteriors'][0].detach().cpu())
                print(f'Pseudo targets: {pseudo_targets}')
                print(f'Noisy predictions: {noisy_predictions}')
                print('\n--\n')

            pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0).to(model.device)
            if pseudo_targets.shape[-1] == 0:
                # skip update — nothing to supervise against
                if online:
                    logits = out['final_posteriors'][-1].detach().cpu()
                    logits = torch.exp(logits)
                    ds_len = logits.shape[-2]
                    ratio = u_len / ds_len
                    overlap_ds = int(overlap / ratio)
                    model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}
                continue

            pseudo_targets = pseudo_targets.repeat(num_negatives, 1)
            augmented_outs = out['final_posteriors'][:num_negatives]

            N, B = augmented_outs.shape[1], augmented_outs.shape[0]
            total_tokens_in_loss = N * B

            loss = ctc_loss_fn(
                augmented_outs.transpose(0, 1),
                pseudo_targets,
                torch.LongTensor([N] * augmented_outs.shape[0]).to(model.device),
                torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0]).to(model.device),
            ) / total_tokens_in_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if online:
                logits = out['final_posteriors'][-1].detach().cpu()
                logits = torch.exp(logits)
                ds_len = logits.shape[-2]
                ratio = u_len / ds_len
                overlap_ds = int(overlap / ratio)
                model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}
        epochs_etime = time.time()
        if print_runtimes:
            print(f'Epoch runtime: {epochs_etime - epochs_stime}')

    if not online:
        model.eval()
        training_data, training_keys = prepare_chunks(spec, seq_len, overlap)
        final_pass_stime = time.time()
        pbar = tqdm(training_keys) if use_tqdm else training_keys
        for i in pbar:
            audio_chunk = training_data[i].clone()
            u_len = audio_chunk.shape[-1]
            audio_chunk = audio_chunk.to(model.device)
            with torch.no_grad():
                out = model(audio_signal=audio_chunk)
            logits = out['final_posteriors'][0].detach().cpu()
            logits = torch.exp(logits)
            ds_len = logits.shape[-2]
            ratio = u_len / ds_len
            overlap_ds = int(overlap / ratio)
            model_outputs[i] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}
        final_pass_etime = time.time()
        if print_runtimes:
            print(f'Final pass runtime: {final_pass_etime - final_pass_stime}')
        model.train()

    logit_position = 0
    for i in sorted(list(model_outputs.keys())):
        logits = model_outputs[i]['logits']
        ds_len = model_outputs[i]['ds_len']
        overlap_ds = model_outputs[i]['overlap_ds']
        logit_position -= overlap_ds if i != 0 else 0
        logit_count[:, logit_position:logit_position + ds_len, :] += 1
        all_logits[:, logit_position:logit_position + ds_len, :] += logits
        logit_position += ds_len

    B, N, C = all_logits.shape
    all_logits = all_logits[logit_count.sum(dim=-1) != 0]
    all_logits = all_logits.reshape(B, -1, C)
    logit_count = logit_count[logit_count.sum(dim=-1) != 0]
    logit_count = logit_count.reshape(B, -1, C)
    logits = all_logits / logit_count
    logits = torch.log(logits)

    if return_params:
        updated_model_params = [p.clone().detach().cpu() for p in model.parameters()]

    # reset model parameters to pre-adaptation state
    for p, p_orig in zip(model.parameters(), original_model_params):
        p.data = p_orig.data.to(p.device)

    if return_params:
        return logits.squeeze(0).numpy(), updated_model_params
    return logits.squeeze(0).numpy()


# Public alias matching the original repo's naming.
dynamic_eval = dynamic_eval_ctc_loss
