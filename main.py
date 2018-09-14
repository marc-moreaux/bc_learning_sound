"""
 Learning from Between-class Examples for Deep Sound Recognition.
 Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada

"""

import sys
import os
import chainer
from os.path import join

import opts
import models
import dataset
from train import Trainer
import cPickle as pickle


def main():
    opt = opts.parse()
    chainer.cuda.get_device_from_id(opt.gpu).use()
    for split in opt.splits:
        print('+-- Split {} --+'.format(split))
        train(opt, split)


def train(opt, split):
    model = getattr(models, opt.netType)(opt.nClasses,
                                         GAP=opt.GAP,
                                         bypass=opt.bypass)
    model.to_gpu()
    optimizer = chainer.optimizers.NesterovAG(lr=opt.LR, momentum=opt.momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(opt.weightDecay))
    train_iter, val_iter = dataset.setup(opt, split)
    trainer = Trainer(model, optimizer, train_iter, val_iter, opt)
    log = {'train_acc': [], 'val_acc': [], 'lr': [], 'train_loss': []}

    if opt.testOnly:
        chainer.serializers.load_npz(join(opt.save,
                                          'model_split{}.npz'.format(split)),
                                     trainer.model)
        val_top1 = trainer.val()
        print('| Val: top1 {:.2f}'.format(val_top1))
        return

    for epoch in range(1, opt.nEpochs + 1):
        train_loss, train_top1 = trainer.train(epoch)
        val_top1 = trainer.val()
        sys.stderr.write('\r\033[K')
        sys.stdout.write(
            '| Epoch: {}/{} | Train: LR {}  Loss {:.3f}  top1 {:.2f} |'
            'Val: top1 {:.2f}\n'.format(epoch, opt.nEpochs,
                                        trainer.optimizer.lr, train_loss,
                                        train_top1, val_top1))
        sys.stdout.flush()
        log['lr'].append(trainer.optimizer.lr)
        log['train_loss'].append(train_loss)
        log['train_acc'].append(train_top1)
        log['val_acc'].append(val_top1)


    if opt.save != 'None':
        # Save weights
        chainer.serializers.save_npz(
            join(opt.save, 'model_split{}.npz'.format(split)), model)
        # Save logs
        with open(join(opt.save, 'logger{}.txt'.format(split)), "w") as f:
            for k, v in log.items():
                f.write(str(k) + ': ' + str(v) + '\n')
        # Save parameters
        with open(join(opt.save, 'opt{}.pkl'.format(split)), "wb") as f:
            pickle.dump(opt, f)


if __name__ == '__main__':
    main()
