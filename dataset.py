import os
import numpy as np
import random
import chainer

import utils as U


class SoundDataset(chainer.dataset.DatasetMixin):
    def __init__(self, sounds, labels, opt, train=True):
        self.base = chainer.datasets.TupleDataset(sounds, labels)
        self.opt = opt
        self.train = train
        self.mix = (opt.BC and train)
        self.preprocess_funcs = self.preprocess_setup()
        self.long_audio = False

    def __len__(self):
        return len(self.base)

    def preprocess_setup(self):
        if self.train:
            funcs = []
            if self.opt.strongAugment:
                funcs += [U.random_scale(1.25)]
            
            funcs += [U.padding(self.opt.inputLength // 2),
                      U.random_crop(self.opt.inputLength),
                      U.normalize(float(2 ** 16 / 2)),  # 16 bit signed
                      ]

        else:
            if self.opt.noiseAugment:
                funcs += [U.add_noise(is_train, 0.15,
                                      self.opt.data,
                                      self.opt.fs,
                                      self.opt.inputLength)]

            funcs = [U.padding(self.opt.inputLength // 2),
                     U.normalize(float(2 ** 16 / 2)),  # 16 bit signed
                     U.multi_crop(self.opt.inputLength, self.opt.nCrops),
                     ]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

    def get_example(self, i):
        if self.mix:  # Training phase of BC learning
            # Select two training examples
            while True:
                sound1, label1 = self.base[random.randint(0, len(self.base) - 1)]
                sound2, label2 = self.base[random.randint(0, len(self.base) - 1)]
                if label1 != label2:
                    break
            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)

            # Mix two examples
            r = np.array(random.random())
            sound = U.mix(sound1, sound2, r, self.opt.fs).astype(np.float32)
            eye = np.eye(self.opt.nClasses)
            label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)
        
        elif self.long_audio:  # Mix two audio on long frame
            sound_len_sec = 10
            sound_len = self.opt.fs * full_sound_len
            sound = np.array(sound_len).astype(np.float32)

            sound1, label1 = self.base[random.randint(0, len(self.base) - 1)]
            sound2, label2 = self.base[random.randint(0, len(self.base) - 1)]
            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)

            # Mix samples
            idx1, idx2 = np.random.randint(0, sound_len - len(sound1), 2)
            sound[idx1: idx1 + len(sound1)] = sound1
            sound[idx2: idx2 + len(sound2)] = sound2
            eye = np.eye(self.opt.nClasses)
            label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)

        else:  # Training phase of standard learning or testing phase
            sound, label = self.base[i]
            sound = self.preprocess(sound).astype(np.float32)
            label = np.array(label, dtype=np.int32)

        if self.train and self.opt.strongAugment:
            sound = U.random_gain(6)(sound).astype(np.float32)

        return sound, label


def setup(opt, split):
    dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.fs // 1000)))

    # Split to train and val
    train_sounds = []
    train_labels = []
    val_sounds = []
    val_labels = []
    for i in range(1, opt.nFolds + 1):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']
        if i == split:
            val_sounds.extend(sounds)
            val_labels.extend(labels)
        else:
            train_sounds.extend(sounds)
            train_labels.extend(labels)

    # Iterator setup
    train_data = SoundDataset(train_sounds, train_labels, opt, train=True)
    val_data = SoundDataset(val_sounds, val_labels, opt, train=False)
    train_iter = chainer.iterators.MultiprocessIterator(train_data, opt.batchSize, repeat=False)
    val_iter = chainer.iterators.SerialIterator(val_data, opt.batchSize // opt.nCrops, repeat=False, shuffle=False)

    return train_iter, val_iter
