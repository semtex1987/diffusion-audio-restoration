# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

import os
import numpy as np 
import json
import argparse
import glob
from subprocess import Popen, PIPE
import yaml
import time 
from datetime import datetime
import shutil
import csv
from tqdm import tqdm
import librosa
import soundfile as sf

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def save_yaml(data, prefix="../configs/temp"):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd_num = np.random.rand()
    rnd_num = rnd_num - rnd_num % 0.000001
    file_name = f"{prefix}_{timestamp}_{rnd_num}.yaml"
    with open(file_name, 'w') as f:
        yaml.dump(data, f)
    return file_name

def shell_run_cmd(cmd):
    print('running:', cmd)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = p.communicate()
    print(stdout)
    print(stderr)

def compute_rolloff_freq(audio_file, roll_percent=0.99):
    """Fallback if no explicit cutoff is provided."""
    y, sr = librosa.load(audio_file, sr=None)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)[0]
    rolloff = int(np.mean(rolloff))
    print('Auto-detected 99 percent rolloff:', rolloff)
    return rolloff

def upsample_one_sample(audio_filename, output_audio_filename, predict_n_steps=50, explicit_cutoff=None):

    assert output_audio_filename != audio_filename, "output filename cannot be input filename"

    inference_config = load_yaml('../configs/inference_files_upsampling.yaml')
    inference_config['data']['predict_filelist'] = [{
        'filepath': audio_filename,
        'output_subdir': '.'
    }]

    # === CRITICAL FIX START ===
    if explicit_cutoff is not None and explicit_cutoff > 0:
        print(f"Using explicit cutoff frequency: {explicit_cutoff} Hz")
        cutoff_freq = int(explicit_cutoff)
    else:
        print("No explicit cutoff provided, attempting to auto-detect...")
        cutoff_freq = compute_rolloff_freq(audio_filename, roll_percent=0.99)
    # === CRITICAL FIX END ===

    inference_config['data']['transforms_aug'][0]['init_args']['upsample_mask_kwargs'] = {
        'min_cutoff_freq': cutoff_freq,
        'max_cutoff_freq': cutoff_freq
    }
    temporary_yaml_file = save_yaml(inference_config)

    # Note: Ensure the path to ensembled_inference_api.py is correct relative to where this runs
    cmd = "cd ../; \
        python ensembled_inference_api.py predict \
            -c configs/ensemble_2split_sampling.yaml \
            -c {} \
            --model.predict_n_steps={} \
            --model.output_audio_filename={}; \
        cd inference/".format(temporary_yaml_file.replace('../', ''), predict_n_steps, output_audio_filename)
    shell_run_cmd(cmd)
    
    if os.path.exists(temporary_yaml_file):
        os.remove(temporary_yaml_file)

def main():
    parser = argparse.ArgumentParser(description='A2SB Upsampler API')
    parser.add_argument('-f','--audio_filename', type=str, help='audio filename to be upsampled', required=True)
    parser.add_argument('-o','--output_audio_filename', type=str, help='path to save upsampled audio', required=True)
    parser.add_argument('-n','--predict_n_steps', type=int, help='number of sampling steps', default=50)
    
    # NEW ARGUMENT
    parser.add_argument('-c','--cutoff', type=float, help='Explicit cutoff frequency in Hz', default=None)
    
    args = parser.parse_args()

    upsample_one_sample(
        audio_filename=args.audio_filename, 
        output_audio_filename=args.output_audio_filename, 
        predict_n_steps=args.predict_n_steps,
        explicit_cutoff=args.cutoff
    )

if __name__ == '__main__':
    main()
