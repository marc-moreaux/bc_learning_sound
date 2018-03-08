#!/usr/bin/python
''' Use this script to produce the Class Activation Mappings

Example :
python ./test.py --save ./results1 --split -1
'''


import models
import dataset
import chainer
from chainer import cuda
from train import Trainer
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import cPickle as pickle


def parse():
    parser = argparse.ArgumentParser(
        description='Perform some tests on a trained model')
    parser.add_argument(
        '--save',
        required=True,
        help='Directory where the model was saved')
    parser.add_argument(
        '--split',
        type=int,
        default=-1,
        help='esc: 1-5, urbansound: 1-10 (-1: run on all splits)')
    args = parser.parse_args()

    # Reformat arguments if necessary
    args.split = [1, 2, 3, 4, 5] if args.split == -1  else [args.split]

    return args


def load_model(save_path, split):
    '''Load a model stored at <save_path> on split <split>
    '''
    # Load opt
    with open(os.path.join(save_path, 'opt{}.pkl'.format(split)), 'rb') as f:
        opt = pickle.load(f)

    # Load model
    model = getattr(models, opt.netType)(opt.nClasses, GAP=opt.GAP)
    chainer.serializers.load_npz(
        os.path.join(opt.save, 'model_split{}.npz'.format(split)), model)
    model.to_gpu()

    return model, opt


def load_first_val_batch(opt, split):
    '''Loads the 1st batch of the validation dataset
    '''
    train_iter, val_iter = dataset.setup(opt, split)
    batch = val_iter.next()
    x_array, t_array = chainer.dataset.concat_examples(batch)
    if opt.nCrops > 1:
        x_array = x_array.reshape((x_array.shape[0] * opt.nCrops, x_array.shape[2]))
    x = chainer.Variable(cuda.to_gpu(x_array[:, None, None, :]))
    t = chainer.Variable(cuda.to_gpu(t_array))

    return x, t


def main():
    args = parse()

    for split in args.split:
        # Load data and model
        model, opt = load_model(args.save, split)
        x, _ = load_first_val_batch(opt, split)

        # Compute CAMs
        y = model(x)
        cams = chainer.cuda.to_cpu(model.maps.data)
        sound = chainer.cuda.to_cpu(x.data)

        # Visualize CAMs
        if opt.GAP:
            for i in range(3):
                viz = cams[i].sum(axis=1)
                fig, axs = plt.subplots(2,1)

                axs[0].plot(sound[i, 0, 0])
                axs[1].plot(viz.T)
                fig.savefig(os.path.join(opt.save, 'split{}_viz{}.png'.format(split, i)))
                fig.clf()


if __name__ == '__main__':
    main()

