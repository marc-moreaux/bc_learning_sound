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
import matplotlib.animation as animation
import numpy as np
import os
import argparse
import cPickle as pickle
import ast
import sys


def fake_parse():
    from argparse import Namespace
    args = Namespace(save='./results_esc10_6',
                     split=[1, ],
                     noiseAugment=False,
                     inputLength=0)
    
    return args


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
    parser.add_argument(
        '--noiseAugment',
        action='store_true',
        help='Add some noise when training')
    parser.add_argument(
        '--inputLength',
        type=int,
        default=0,
        help='change prefered input size')
    args = parser.parse_args()

    # Reformat arguments if necessary
    args.split = [1, 2, 3, 4, 5] if args.split == -1  else [args.split]

    return args


def get_class_names(opt):
    '''Get a correspondance between class number and class_name
    '''
    import pandas as pd
    df = pd.read_csv(os.path.join(opt.data, 'esc50/ESC-50-master/meta/esc50.csv'))
    if opt.dataset == 'esc10' or opt.dataset == 'esc50':
        class_names = dict(set(zip(df.target, df.category)))
        if opt.dataset == 'esc10':
            esc10_classes = [0, 10, 11, 20, 38, 21, 40, 41, 1, 12]
            class_names = {i: class_names[c_idx] for i, c_idx in enumerate(esc10_classes)}
    
    return class_names


def fix_opt(opt):
    if 'results_' in opt.save:
        opt.save = opt.save.replace('results_', 'results/')
    if not 'noiseAugment' in opt:
        opt.noiseAugment = False
    return opt


def change_opt_wrt_args(opt, args):
    '''Change opt according to args parameters
    '''
    opt.noiseAugment = args.noiseAugment
    if args.inputLength > 0:
        opt.inputLength = args.inputLength
    return opt


def load_model(save_path, split):
    '''Load a model stored at <save_path> on split <split>
    '''
    # Load opt
    with open(os.path.join(save_path, 'opt{}.pkl'.format(split)), 'rb') as f:
        opt = pickle.load(f)
        opt = fix_opt(opt)

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
    x_array, lbls = chainer.dataset.concat_examples(batch)
    if opt.nCrops > 1:
        x_array = x_array.reshape((x_array.shape[0] * opt.nCrops, x_array.shape[2]))
    x = chainer.Variable(cuda.to_gpu(x_array[:, None, None, :]))

    return x, lbls


def plot_CAM_visualizations(sounds, cams, lbls, split, opt):
    '''save the visualization of a CAM in a .png
    '''
    class_names = get_class_names(opt)

    for i in range(3):
        viz = cams[i].sum(axis=1)
        fig, axs = plt.subplots(2, 1, figsize=(15, 9))
        axs[0].set_title(class_names[lbls[i/opt.nCrops]])
        axs[0].plot(sounds[i, 0, 0])
        for _i in range(opt.nClasses):
            axs[1].plot(viz[_i], label=class_names[_i])

        axs[1].legend(ncol=5, bbox_to_anchor=(0., 1.02, 1., .102), loc=3)
        title = '{}, {}'.format(class_names[viz.max(axis=1).argmax()],
                                class_names[viz.mean(axis=1).argmax()])
        axs[1].set_title(title, loc='right')

        save_path = os.path.join(opt.save, 'split{}_viz{}.png'.format(split, i))
        fig.savefig(save_path, dpi=100)
        fig.clf()


def plot_learning(log_path, split, opt):
    '''Reads logs and display the learning curves
    '''
    with open(log_path, "r") as f:
        logs = {}
        for line in f:
            k, v = line.split(': ')
            v = ast.literal_eval(v)
            logs[k] = v
    
    fig, axs = plt.subplots(2, 1, figsize=(15, 12))
    end_mean = np.array(logs['val_acc'][-20:]).mean()
    axs[0].set_title(str(end_mean))
    axs[0].plot(logs['train_acc'])
    axs[0].plot(logs['val_acc'])
    axs[0].set_ylim([4, 30])
    axs[1].plot(logs['train_loss'])
    save_path = os.path.join(opt.save, 'learning_split{}.png'.format(split))
    fig.savefig(save_path, dpi=100)
    fig.clf()


def plot_training_waves(opt, split):
    train_iter, val_iter = dataset.setup(opt, split)
    batch = train_iter.next()
    class_names = get_class_names(opt)

    fig, axs = plt.subplots(8, 8, figsize=(15, 12))
    for idx in range(8 * 8):
        sample = batch[idx]
        ax = axs[idx / 8][idx % 8]
        ax.plot(sample[0])
        ax.set_title(class_names[sample[1].argmax()])
        ax.set_ylim(-1, 1)
        ax.set_axis_off()
    save_path = os.path.join(opt.save, 'samples_split{}.png'.format(split))
    fig.savefig(save_path, dpi=100)
    fig.clf()


def main():
    if len(sys.argv) > 1:
        args = parse()
    else:
        args = fake_parse()

    for split in args.split:
        # Load data and model
        model, opt = load_model(args.save, split)
        opt = change_opt_wrt_args(opt, args)
        x, lbls = load_first_val_batch(opt, split)

        # Plot training samples
        plot_training_waves(opt, split)

        # Compute CAMs
        y = model(x)
        cams = chainer.cuda.to_cpu(model.maps.data)
        sounds = chainer.cuda.to_cpu(x.data)

        # Visualize CAMs
        if opt.GAP:
            plot_CAM_visualizations(sounds, cams, lbls, split, opt)
        
        # Visualize learning
        log_path = os.path.join(opt.save, 'logger{}.txt'.format(split))
        plot_learning(log_path, split, opt)
        

if __name__ == '__main__':
    main()

