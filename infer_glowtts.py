import matplotlib.pyplot as plt
import sys
import librosa
import numpy as np
import os
import glob
import json
import soundfile as sf
import torch
from text import text_to_sequence, cmudict
from text.symbols import symbols
from text.marathi_symbols import marathi_symbols
import commons
import attentions
import modules
import models
import utils
import librosa
import argparse
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

MAX_WAV_VALUE = 32768.0

def inference(vocoder,
            vocoder_path,
            text,
            lang):

    if vocoder == 'waveglow':

        if vocoder_path == None:
            raise Exception('provide path to vocoder checkpoint with voc_path aruement')
        # waveglow_path = os.path.join(args.voc_path, args.version)
        waveglow_path = args.voc_path
        existing_checkpoints = [int(f.replace('waveglow_', '')) for f in os.listdir(waveglow_path) if f.startswith('waveglow')]
        if len(existing_checkpoints)>0:
            latest_checkpoint = os.path.join(waveglow_path, 'waveglow_'+str(max(existing_checkpoints)))
        print(latest_checkpoint)
        waveglow = torch.load(latest_checkpoint)['model']
        waveglow = waveglow.remove_weightnorm(waveglow)
        _ = waveglow.cuda().eval()
        # from apex import amp
        # waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")
        # exit()
    elif vocoder != 'gl':
        raise Exception('Only waveglow and griffin lim (gl) supported')

    model_dir = os.path.join(f'./chk/glowtts/{lang}')
    hps = utils.get_hparams_from_dir(model_dir)
    checkpoint_path = os.path.join(model_dir, f'{lang}.pth')
    if lang == 'mr':
        symbols_ = marathi_symbols
    model = models.FlowGenerator(
        len(symbols_) + getattr(hps.data, "add_blank", False),
        out_channels=hps.data.n_mel_channels,
        **hps.model).to("cuda")

    utils.load_checkpoint(checkpoint_path, model)
    model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
    _ = model.eval()


    def normalize_audio(x, max_wav_value=hps.data.max_wav_value):
        return np.clip((x / np.abs(x).max()), -1, 1)
    
    tst_stn = text

    cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)
    if getattr(hps.data, "add_blank", False):
        print(tst_stn.strip())

        text_norm = text_to_sequence(tst_stn.strip(), ['hindi_cleaners'], cmu_dict)
        text_norm = commons.intersperse(text_norm, len(symbols_))
    else: # If not using "add_blank" option during training, adding spaces at the beginning and the end of utterance improves quality
        tst_stn = " " + tst_stn.strip() + " "
        text_norm = text_to_sequence(tst_stn.strip(), ['hindi_cleaners'])

    sequence = np.array(text_norm)[None, :]
    # print("".join([symbols[c] if c < len(symbols) else "<BNK>" for c in sequence[0]]))
    x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()
    print(x_tst)
    with torch.no_grad():
        noise_scale = .667
        length_scale = 1.0
        (y_gen_tst, *_), *_, (attn_gen, *_) = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)
        if vocoder == 'waveglow':
            args.message='glowtts_wg'+add_to_message
            try:
                audio = waveglow.infer(y_gen_tst.half(), sigma=.666)
            except:
                audio = waveglow.infer(y_gen_tst, sigma=.666)
            audio = audio * MAX_WAV_VALUE
            audio = audio.squeeze()
            audio = audio.cpu().numpy()
            audio = audio.astype('int16')
        elif vocoder == 'gl':
            y_gen_tst = y_gen_tst.squeeze().cpu().numpy()
            y_gen_tst = np.exp(y_gen_tst)
            S = librosa.feature.inverse.mel_to_stft(
                    y_gen_tst,
                    power=1,
                    sr=22050,
                    n_fft=1024,
                    fmin=0,
                    fmax=8000.0)
            audio = librosa.core.griffinlim(
                    S,
                    n_iter=32,
                    hop_length=256,
                    win_length=1024)
            # audio = torch.from_numpy(audio)
            audio = audio * MAX_WAV_VALUE
            audio = audio.astype('int16')
            # audio = normalize_audio(audio.clamp(-1,1).data.cpu().float().numpy())
    
    
    sf.write('sample.wav', audio, 22050)
