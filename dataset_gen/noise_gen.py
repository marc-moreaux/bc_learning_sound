"""
 Dataset preparation code for noise some selected noise
 Usage: python noise_gen.py [path]
 FFmpeg should be installed.

"""

import sys
import os
import subprocess

from esc_gen import convert_fs
import numpy as np
import wavio


def create_dataset(dir_path, dest_dir, fs):
    print('* Creating the .npz in {}'.format(dir_path))
    wav_dataset = {'train': [], 'valid': []}

    for wav_file in sorted(os.listdir(dir_path)):
        wav_file = os.path.join(dir_path, wav_file)
        sound = wavio.read(wav_file).data.T[0]
        start = sound.nonzero()[0].min()
        end = sound.nonzero()[0].max()
        sound = sound[start: end + 1]  # Remove silent sections
        split = len(sound) * 70 / 100
        train_sound = sound[: split]
        valid_sound = sound[split:]
        wav_dataset['train'].append(train_sound)
        wav_dataset['valid'].append(valid_sound)

    npz_file_path = os.path.join(dest_dir, 'wav{}.npz'.format(fs // 1000))
    np.savez(npz_file_path, **wav_dataset)


noise_dir = os.path.join(sys.argv[1], 'noise')
wav_dir_orig = os.path.join(noise_dir, 'to_export')
fs_list = [16000, 44100]

for fs in fs_list:
    wav_dir_fs = os.path.join(noise_dir, 'wav{}'.format(fs // 1000))
    convert_fs(wav_dir_orig, wav_dir_fs, fs)

for fs in fs_list:
    wav_dir_fs = os.path.join(noise_dir, 'wav{}'.format(fs // 1000))
    create_dataset(wav_dir_fs, noise_dir, fs)
