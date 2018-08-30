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
import itertools
import cPickle as pickle
import ast
import sys
from sklearn.metrics import confusion_matrix


# General loading functions
def fake_parse():
    from argparse import Namespace
    args = Namespace(save='./results/esc10_att_7b',
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
    args.split = [1, 2, 3, 4, 5] if args.split == -1 else [args.split]

    return args


def get_class_names(opt, add_void=False):
    '''Get a correspondance between class number and class_name
    '''
    import pandas as pd
    df = pd.read_csv(os.path.join(opt.data, 'esc50/ESC-50-master/meta/esc50.csv'))
    if opt.dataset == 'esc10' or opt.dataset == 'esc50':
        class_names = dict(set(zip(df.target, df.category)))
        if opt.dataset == 'esc10':
            esc10_classes = [0, 10, 11, 20, 38, 21, 40, 41, 1, 12]
            class_names = {i: class_names[c_idx] for i, c_idx in enumerate(esc10_classes)}
    
    if add_void:
        class_names[-1] = 'silence'

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


# Model and sample loading
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


def val_batch_gen(opt, split, remove_padding=False):
    train_iter, val_iter = dataset.setup(opt, split)
    
    for batch in val_iter:
        x_array, lbls = chainer.dataset.concat_examples(batch)
        if opt.nCrops > 1 and opt.longAudio == 0:
            x_array = x_array.reshape((x_array.shape[0] * opt.nCrops, x_array.shape[2]))
        xs = chainer.Variable(cuda.to_gpu(x_array[:, None, None, :]))

        if remove_padding:
            xs = xs[:, :, :, opt.inputLength // 2 : - opt.inputLength // 2]
            lbls = lbls[:, opt.inputLength // 2 : - opt.inputLength // 2]

        yield xs, lbls


# Localisation functions
def get_active_intervals(mask):
    '''On a 0/1 signal, returns idxs of intervals with one
    '''
    tmp1 = mask * 1
    tmp2 = np.pad(mask, 1, 'constant')[:-2] * 1
    start_idx = np.where((tmp1 - tmp2) == 1)[0]
    end_idx = np.where((tmp1 - tmp2) == -1)[0]
    if tmp1[-1] == 1:
        end_idx = np.concatenate([end_idx, [len(tmp1),]])

    return zip(start_idx, end_idx)


def shrinked_labels_for_loc(lbls, window_size=20570, window_step=3072):
    '''
    Reduces size of lbls to the same size as GAP layer
    params
        window_size: size of a Envnet window (computed going up the conv stack)
        window_step: size of a Envnet step (space btw each GAP points)
        Envnet10: wsi=8174, wst=1440
        Envnet50: wsi=20570, wst=3072
    '''
    window_size = window_size - window_step + 1
    print(lbls.shape, window_size/2)
    gts = lbls[:, window_size/2 : -window_size/2 : window_step]
    return np.array(gts)


def scale_pred_at_fs(pred, lbl_len, window_size=20570, window_step=3072):
    window_size = window_size - window_step + 1
    scaled_pred = np.zeros(lbl_len) - 1
    scaled_pred[:window_size/2 + window_step/2] = pred[0]
    scaled_pred[-window_size/2 - window_step/2:] = pred[-1]
    for i in range(len(pred)):
        start = window_size/2 + i * window_step - window_step/2
        end = window_size/2 + i * window_step + window_step/2
        scaled_pred[start : end] = pred[i]

    return scaled_pred


def get_localisation_prediction(cam,
                                act_thrld=30,
                                act_window=10,
                                min_act_per_window=3,
                                zero_activations=None):
    '''Retrieve localisation of filtered masks and compute CAM on that zone
    returns a vector (lengnt = len(cam)) filled with '-1' when nothing is seen and the 
    class predicted elsewhere
    '''
    viz = cam.mean(axis=1)
    if zero_activations is not None:
        viz = (viz.T - zero_activations).T
    
    c_mask = viz > act_thrld
    c_mask2 = np.array([np.convolve(c_mask[c], np.ones(act_window), mode='same')
                        for c in range(len(cam))])
    predictions = c_mask2.argmax(axis=0) * (c_mask2.max(axis=0) >= min_act_per_window)
    predictions = predictions + (c_mask2.max(axis=0) >= min_act_per_window) - 1

    return predictions


def evaluate_localisation(cams, lbls,
                          act_thrld=30,
                          act_window=10,
                          min_act_per_window=3,
                          zero_activations=None):
    #gts = shrinked_labels_for_loc(lbls)
    gts = lbls

    metrics = []
    for i in range(len(lbls)):
        gt = gts[i]
        cam = cams[i]
        pred = get_localisation_prediction(
            cam,
            act_thrld=act_thrld,
            act_window=act_window,
            min_act_per_window=min_act_per_window,
            zero_activations=zero_activations)
        pred = scale_pred_at_fs(pred, len(gt))

        FN = (pred[gt != pred] == -1).sum() / float(len(pred))  # gt isn't met by pred
        TN = (gt[gt == pred] == -1).sum() / float(len(pred))  # gt and pred are negatives
        TP = (pred[gt == pred] != -1).sum() / float(len(pred))  # stg predicted is good
        FP = (pred[gt != pred] != -1).sum() / float(len(pred))  # stg predicted isn't good
        conf_matrix = confusion_matrix(gt, pred, range(-1,50))
        ommission = pred[gt != -1] == -1
        ommission = pred[gt != -1] != gt[gt != -1]
        insertion = pred[pred != -1] != gt[pred != -1]
        accuracy = gt == pred
        ommission = ommission.mean() if len(ommission) > 0 else 0
        insertion = insertion.mean() if len(insertion) > 0 else 0
        accuracy = accuracy.mean() if len(accuracy) > 0 else 0
        metrics.append((ommission, insertion, accuracy, FN, TP, FP, TN, conf_matrix))
    
    return metrics


# Plotting functions
def plot_CAM_visualizations(sounds,
                            cams,
                            lbls,
                            split,
                            opt,
                            on_screen,
                            act_thrld,
                            act_window,
                            min_act_per_window,
                            zero_activations=None):
    '''save the visualization of a CAM in a .png
    '''
    class_names = get_class_names(opt)
    gts = shrinked_labels_for_loc(lbls)
    gts = lbls
    for i in range(5):
        viz = cams[i].mean(axis=1)
        if zero_activations is not None:
            viz = (viz.T - zero_activations).T
        fig, axs = plt.subplots(2, 1, figsize=(15, 6))

        # Set title
        if len(lbls.shape) == 1:
            ttl = class_names[lbls[i / opt.nCrops]]
        if len(lbls.shape) == 2:
            _lbl = list(set(lbls[i]))
            _lbl = [int(j) for j in _lbl if j >= 0]
            ttl = '{} and {}'.format(
                class_names[_lbl[0]],
                class_names[_lbl[1]] )
        
        # Plot sound
        pred = get_localisation_prediction(cams[i],
                                           act_thrld=act_thrld,
                                           act_window=act_window,
                                           min_act_per_window=min_act_per_window,
                                           zero_activations=zero_activations)
        pred = scale_pred_at_fs(pred, len(gts[0]))        
        axs[0].set_title(ttl)
        axs[0].plot(sounds[i, 0, 0])
        axs[0].plot(pred/10. - 2.5)
        axs[0].plot(lbls[i]/10. - 2.5)
        axs[0].set_ylim(-2.6, 2.6)
        axs[0].set_axis_off()
        
        # Plot cams
        for _i in range(opt.nClasses):
            axs[1].plot(viz[_i], label=class_names[_i])
        axs[1].legend(ncol=5, loc=2)
        # title = '{}, {}'.format(class_names[viz.max(axis=1).argmax()],
        #                         class_names[viz.mean(axis=1).argmax()])
        # axs[1].set_title(title, loc='right')

        _noisy = '_n' if opt.noiseAugment else ''
        save_path = os.path.join(opt.save, 'split{}_viz{}{}.png'.format(split, i, _noisy))
        if on_screen:
            plt.show()
        else:
            fig.savefig(save_path, dpi=300)
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


def plot_confusion_matrix(cm, classes, opt, split,
                          normalize=False,
                          cmap=plt.cm.Blues,
                          on_screen=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') * 100 / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(1, 1, figsize=(22, 22))
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '1.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    _noisy = '_n' if opt.noiseAugment else ''
    save_path = os.path.join(opt.save, 'split{}_cm{}.png'.format(split, _noisy))

    if on_screen:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
        plt.clf()


def null_audio_activations(model):
    zeros = np.zeros((1,1,1,100000), dtype=np.float32)
    zeros = chainer.Variable(cuda.to_gpu(zeros))
    y = model(zeros)
    cams = chainer.cuda.to_cpu(model.maps.data)
    zero_activations = cams[0].mean(axis=1)[:, 0]

    return zero_activations


def main():
    if len(sys.argv) > 1:
        args = parse()
    else:
        args = fake_parse()

    for split in args.split:
        # Load data and model
        model, opt = load_model(args.save, split)
        opt = change_opt_wrt_args(opt, args)

        # Plot training samples
        plot_training_waves(opt, split)

        # Compute CAMs
        for _ in range(2):
            opt.noiseAugment = not opt.noiseAugment
            x, lbls = load_first_val_batch(opt, split)
            y = model(x)
            
            if opt.GAP:
                try:
                    cams = chainer.cuda.to_cpu(model.maps.data)
                    sounds = chainer.cuda.to_cpu(x.data)
                    plot_CAM_visualizations(sounds, cams, lbls, split, opt, False)
                except:
                    print 'CAMS part failed for {}'.format(args.save)

        # Visualize learning
        log_path = os.path.join(opt.save, 'logger{}.txt'.format(split))
        plot_learning(log_path, split, opt)


def main2():
    '''Evaluate localization
    '''
    import utils; reload(utils)
    import dataset; reload(dataset)
    if len(sys.argv) > 1:
        args = parse()
    else:
        args = fake_parse()

    # Iterate throw the learned models
    a_metrics = []
    for split in args.split:
        # Load data and model
        model, opt = load_model(args.save, split)
        zero_activations = null_audio_activations(model)
        zero_activations = None 
        opt = change_opt_wrt_args(opt, args)
        opt.noiseAugment = True
        opt.batchSize = 16
        opt.inputTime = 2.5
        opt.longAudio = 8

        # Compute CAMs
        metrics = []
        for _ in range(100):
            val_batch = val_batch_gen(opt, split, remove_padding=False)
            x, lbls = val_batch.next()
            y = model(x)
            
            cams = chainer.cuda.to_cpu(model.maps.data)
            metrics.extend(evaluate_localisation(cams,
                                                 lbls,
                                                 act_thrld=7,
                                                 act_window=7,
                                                 min_act_per_window=3,
                                                 zero_activations=None))
        
        a_metrics.extend(metrics)
        ommission, insertion, acc, FN, TP, FP, TN, conf_matrix = map(np.array, zip(*metrics))
        conf_matrix = conf_matrix.sum(axis=0)
        print("ommission {}, insertion {}, acc {}, omm & ins {}, FN {}, TP {}, FP {}, TN {}",
              ommission.mean(), insertion.mean(), acc.mean(),
              (ommission + insertion).mean(),
              FN.mean(), TP.mean(), FP.mean(), TN.mean())

        c_names = get_class_names(opt, add_void=True)
        c_names = [c_names[k] for k in sorted(c_names.keys())]
        plot_confusion_matrix(conf_matrix, c_names, opt,
                              split, normalize=True, on_screen=False)
        sounds = chainer.cuda.to_cpu(x.data)
        plot_CAM_visualizations(sounds,
                                cams,
                                lbls,
                                split,
                                opt,
                                False,
                                act_thrld=7,
                                act_window=7,
                                min_act_per_window=3,
                                zero_activations=None)


if __name__ == '__main__':
    main2()
    pass

