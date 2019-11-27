"""
 Dataset preparation code for ESC-50 and ESC-10 [Piczak, 2015].
 Usage: python esc_gen.py --save [path]
 FFmpeg should be installed.

"""

import sys
import os
import subprocess
import shutil
import argparse

import glob
import numpy as np
import wavio


def parse():
    parser = argparse.ArgumentParser(description='Dataset parameters parser')

    parser.add_argument('--save', required=True, help='The datapath')
    parser.add_argument('--DEBUG', action='store_false', help='debug mode')
    parser.add_argument('--threshold_sound', type=float, default=0, help='sound threshold')

    args = parser.parse_args()
    return args


def main():
    args = parse()
    esc50_path = os.path.join(args.save, 'esc50')
    esc10_path = os.path.join(args.save, 'esc10')
    if not os.path.isdir(esc50_path):
        os.mkdir(esc50_path)
    if not os.path.isdir(esc10_path):
        os.mkdir(esc10_path)
    fs_list = [16000, 44100]  # EnvNet and EnvNet-v2, respectively

    # Download ESC-50
    if not os.path.isdir(os.path.join(esc50_path, 'ESC-50-master')):
        subprocess.call('wget -P {} https://github.com/karoldvl/ESC-50/archive/master.zip'.format(
            esc50_path), shell=True)
        subprocess.call('unzip -d {} {}'.format(
            esc50_path, os.path.join(esc50_path, 'master.zip')), shell=True)
        os.remove(os.path.join(esc50_path, 'master.zip'))

    # Convert sampling rate
    for fs in fs_list:
        if fs == 44100:
            continue
        else:
            convert_fs(os.path.join(esc50_path, 'ESC-50-master', 'audio'),
                       os.path.join(esc50_path, 'wav{}'.format(fs // 1000)),
                       fs)

    # Create npz files
    for fs in fs_list:
        if fs == 44100:
            src_path = os.path.join(esc50_path, 'ESC-50-master', 'audio')
        else:
            src_path = os.path.join(esc50_path, 'wav{}'.format(fs // 1000))

        create_dataset(src_path,
                       os.path.join(esc50_path, 'wav{}.npz'.format(fs // 1000)),
                       os.path.join(esc10_path, 'wav{}.npz'.format(fs // 1000)),
                       fs,
                       args)


def convert_fs(src_path, dst_path, fs):
    print(('* {} -> {}'.format(src_path, dst_path)))
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    for src_file in sorted(glob.glob(os.path.join(src_path, '*.wav'))):
        if not os.path.isfile(src_file):
            dst_file = src_file.replace(src_path, dst_path)
            subprocess.call('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(
                src_file, fs, dst_file), shell=True)


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


def create_dataset(src_path, esc50_dst_path, esc10_dst_path, fs, args):
    print(('* {} -> {}'.format(src_path, esc50_dst_path)))
    print(('* {} -> {}'.format(src_path, esc10_dst_path)))
    esc10_classes = [0, 10, 11, 20, 38, 21, 40, 41, 1, 12]  # ESC-10 is a subset of ESC-50
    esc50_dataset = {}
    esc10_dataset = {}

    for fold in range(1, 6):
        esc50_dataset['fold{}'.format(fold)] = {}
        esc50_sounds = []
        esc50_labels = []
        esc10_dataset['fold{}'.format(fold)] = {}
        esc10_sounds = []
        esc10_labels = []

        for wav_file in sorted(glob.glob(os.path.join(src_path, '{}-*.wav'.format(fold)))):
            sound = wavio.read(wav_file).data.T[0]
            if args.threshold_sound == 0:  # Remove silent sections
                start = sound.nonzero()[0].min()
                end = sound.nonzero()[0].max()
                sound = sound[start: end + 1]
            else:
                sound = filter_silent_audio(sound,
                                            fs,
                                            m_section_engy_thr=args.threshold_sound,
                                            is_debug=args.DEBUG)
            label = int(os.path.splitext(wav_file)[0].split('-')[-1])
            esc50_sounds.append(sound)
            esc50_labels.append(label)
            if label in esc10_classes:
                esc10_sounds.append(sound)
                esc10_labels.append(esc10_classes.index(label))

        esc50_dataset['fold{}'.format(fold)]['sounds'] = esc50_sounds
        esc50_dataset['fold{}'.format(fold)]['labels'] = esc50_labels
        esc10_dataset['fold{}'.format(fold)]['sounds'] = esc10_sounds
        esc10_dataset['fold{}'.format(fold)]['labels'] = esc10_labels

    np.savez(esc50_dst_path, **esc50_dataset)
    np.savez(esc10_dst_path, **esc10_dataset)


if __name__ == '__main__':
    main()
