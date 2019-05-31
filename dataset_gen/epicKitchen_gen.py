#!/usr/bin/python3
"""
 Dataset preparation code for epicKitchen
 Usage: python esc_gen.py --DB_path [path]
 FFmpeg should be installed.

"""

import sys
import os
import subprocess
import shutil
import argparse
from os.path import join

import glob
import numpy as np
import pandas as pd
import wavio


def parse():
    parser = argparse.ArgumentParser(description='Dataset parameters parser')

    parser.add_argument('--DB_path', required=True,
                        help='The datapath of the dataset')
    parser.add_argument('--DEBUG', action='store_false',
                        help='debug mode')
    parser.add_argument('--threshold_sound', type=float, default=0,
                        help='sound threshold')
    parser.add_argument('--csv_path', type=str,
                        help='path of csv with annotations')

    args = parser.parse_args()
    return args


def main():
    args = parse()
    epicKitchen_path = args.DB_path
    if not os.path.isdir(epicKitchen_path):
        os.mkdir(epicKitchen_path)
    fs_list = [16000, 44100]  # EnvNet and EnvNet-v2, respectively

    # Download epicKitchen
    # TODO

    # Convert sampling rate
    for fs in []: #fs_list:
        if fs == 44100:
            convert_fs(join(epicKitchen_path),
                       join(epicKitchen_path, 'wav{}'.format(fs // 1000)),
                       fs,
                       args)
        else: # fs == 16000
            convert_fs(join(epicKitchen_path),
                       join(epicKitchen_path, 'wav{}'.format(fs // 1000)),
                       fs,
                       args)

    # Create npz files
    for fs in fs_list:
        if fs == 44100:
            src_path = join(epicKitchen_path, 'wav{}'.format(fs // 1000))
        else:
            src_path = join(epicKitchen_path, 'wav{}'.format(fs // 1000))

        create_dataset(src_path,
                       join(epicKitchen_path, 'wav{}.npz'.format(fs // 1000)),
                       fs,
                       args)


def convert_fs(src_path, dst_path, fs, args):
    # Print info
    print('\nConvert fs: \n {} -> {}'.format(src_path, dst_path))

    # Init some var
    df = pd.read_csv(args.csv_path)
    df = df[df.classId >= 0]
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)

    # loop through dataframe and extract audio accordingly
    for idx, fileInfo in df.iterrows():
        # Get file path corresponding to current row
        fName = fileInfo.fileName
        src_file = glob.glob(join(src_path, '**/{}*'.format(fName)),
                             recursive=True)[0]

        # Maybe change file name
        newfName = fName
        start = None
        if 'start_timestamp' in df.columns:
            newfName = '{}_{}'.format(fName, fileInfo.start_timestamp)
            start, delta = get_start_delta(fileInfo)

        # Call ffmpeg to transform file
        dst_file = join(dst_path, '{}.wav'.format(newfName))
        if os.path.isfile(src_file) and not os.path.isfile(dst_file):
            if start is not None:
                cmd = 'ffmpeg -y -ss {} -i "{}" -t {} \
                        -ac 1 -ar {} -vn "{}"'.format(
                    str(start), src_file,
                    str(delta), fs, dst_file)
            else:
                cmd = 'ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(
                        src_file, fs, dst_file)
            subprocess.call(cmd, shell=True)


def get_start_delta(row):
    import datetime
    start = str_to_timedelta(row.start_timestamp)
    stop = str_to_timedelta(row.stop_timestamp)
    delta = (stop - start).total_seconds()
    if delta < 1:
        delta = datetime.timedelta(0, 1)

    return start, delta


def str_to_timedelta(mStr):
    '''gets a time like "00:12:23.23" and outputs a datatime.time
    '''
    from datetime import time, timedelta
    mStr = mStr.replace('.', ':').split(':')
    mTime = time(*map(int, mStr))
    mTime = timedelta(hours=mTime.hour,
                      minutes=mTime.minute,
                      seconds=mTime.second,
                      microseconds=mTime.microsecond * 1e4)
    return mTime


def filter_silent_audio(sound,
                        fs,
                        m_section_t=0.010,
                        m_section_engy_thr=0.0005,
                        n_micro_section_thr=3,
                        M_frame_attention_size=1.1,
                        is_debug=False):
    '''Compute energy of micro-sections (of 10ms) and remove sound when
    large section mostly doesn't have energy
    
    sound: raw sound
    fs: audio sampling frequency
    m_section_t: time of a micro-section in ms
    m_section_engy: threshold energy of a micro section (mean of squares)
    n_micro_section_thr: minimum amount of active micro-section to have to consider a sound
    M_frame_attention_size: macro-section's frame of attention (in sec.)
    '''
    
    def _micro_section_energy_over_thr(micro_section):
        micro_section = micro_section / 25000.
        _sum = (micro_section ** 2).mean()
        if _sum < m_section_engy_thr:
            return False
        return True

    n_to_keep = 0
    # Check energy of micro-sections
    while n_to_keep == 0:
        m_section_len = int(m_section_t * fs)
        engy_over_thr = []
        for i in range(0, len(sound) - m_section_len, m_section_len):
            start = i
            end = i + m_section_len
            micro_section = sound[start: end]
            engy_over_thr.append(
                _micro_section_energy_over_thr(micro_section))
        engy_over_thr = np.array(engy_over_thr)

        # Search for high energy Macro_sections (of M_section_len pts)
        # True if it has <n_micro_section_thr> micro-sections active
        M_frame_attention_size_pts = M_frame_attention_size * fs
        M_section_len = int(M_frame_attention_size_pts / m_section_len)
        to_keep = []
        for i in range(0, len(engy_over_thr) - M_section_len):
            pad = m_section_len  # pad start and end of array
            if i == 0 or i == len(engy_over_thr) - M_section_len - 1:
                pad = (M_section_len / 2 + 1) * m_section_len
            if (engy_over_thr[i: i + M_section_len]).sum() > n_micro_section_thr:
                    to_keep.extend([True,] * pad)
            else :
                    to_keep.extend([False,] * pad)
        to_keep = np.array(to_keep)

        # Only keep active sections
        n_to_keep = int(to_keep.sum())
        if n_to_keep == 0:  # increase sound if nothing is detected
            sound = sound.astype(float) * 1.2

    new_sound = sound.copy() * 0
    sound_to_keep = sound[: len(to_keep)][to_keep]
    for i in range(0, len(new_sound) - n_to_keep, n_to_keep):
        new_sound[i: i + n_to_keep] = sound_to_keep
    new_sound[i + n_to_keep:] = sound_to_keep[: len(new_sound) - i - n_to_keep]

    if is_debug == True:
        import matplotlib.pyplot as plt
        import sounddevice as sd
        sd.play(sound, fs)
        fig, axs = plt.subplots(4, 1)
        axs[0].plot(sound)
        axs[1].plot(engy_over_thr)
        axs[2].plot(to_keep)
        axs[3].plot(new_sound)
        plt.show()

    return sound


def create_dataset(src_path, epicKitchen_dst_path, fs, args):
    print('\nCreate dataset : \n* {} -> {}'.format(
        src_path, epicKitchen_dst_path))

    # Initialize some variables
    epicKitchen_dataset = {}
    df = pd.read_csv(args.csv_path)
    df = df[df.classId >= 0]

    for fold in range(1, 6):
        print('fold {}'.format(fold))
        epicKitchen_dataset['fold{}'.format(fold)] = {}
        epicKitchen_sounds = []
        epicKitchen_labels = []
        df_fold = df[df.fold + 1 == fold]

        # Loop through fold files
        for idx, fileInfo in df_fold.iterrows():
            # Maybe change file name
            fName = fileInfo.fileName
            newfName = fName
            start = None
            if 'start_timestamp' in df.columns:
                newfName = '{}_{}'.format(fName, fileInfo.start_timestamp)

            # Get file path corresponding to newfName
            fName = fileInfo.fileName
            wav_file = glob.glob(join(src_path, '**/{}*'.format(newfName)),
                                 recursive=True)[0]

            # Process sound
            sound = wavio.read(wav_file).data.T[0]
            if args.threshold_sound == 0:  # Remove silent sections
                start = sound.nonzero()[0].min()
                end = sound.nonzero()[0].max()
                sound = sound[start: end + 1]
            else:
                sound = filter_silent_audio(
                    sound,
                    fs,
                    m_section_engy_thr=args.threshold_sound,
                    is_debug=args.DEBUG)

            label = fileInfo.classId
            epicKitchen_sounds.append(sound)
            epicKitchen_labels.append(label)

        epicKitchen_dataset['fold{}'.format(fold)]['sounds'] = epicKitchen_sounds
        epicKitchen_dataset['fold{}'.format(fold)]['labels'] = epicKitchen_labels

    np.savez(epicKitchen_dst_path, **epicKitchen_dataset)


if __name__ == '__main__':
    main()
