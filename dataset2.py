#%%
import tensorflow as tf
import os
import itertools
import numpy as np
import utils
import utils as U
#reload(utils)
U = utils

Dataset = tf.data.Dataset 
Iterator = tf.data.Iterator


def get_split(opt, split):
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
    
    return (train_sounds, train_labels), (val_sounds, val_labels)


args = U.fake_parse()
opt = U.load_opt(args.save, 1)
dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.fs // 1000)), encoding="latin1")
mix = opt.BC


# Split train and val
train = []
val = []
for _s in range(5):
    _train, _val = get_split(opt, _s)
    for _ in range(3):
        _train[0].extend(_train[0])
        _train[1].extend(_train[1])
        _val[0].extend(_val[0])
        _val[1].extend(_val[1])
    train.append(_train)
    val.append(_val)

# train = tuple(train)
# val = tuple(val)


def dataset_input_fn(is_train, batch_size=64, split=1):
    sounds, labels = train[split-1] if is_train is True else val[split-1]
    labels = np.array(labels).reshape((-1, 1))
    dataset = tf.data.Dataset.from_generator(
        lambda: zip(sounds, labels),
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape([None]),
                       tf.TensorShape(1)))

    # if is_train:
    # if opt.strongAugment:
    #     dataset = dataset.map(U.random_scale(1.25))
    dataset = dataset.map(U.padding(opt.inputLength // 2))
    dataset = dataset.map(U.random_crop(opt.inputLength))
    dataset = dataset.map(U.normalize(float(2 ** 16 / 2)))
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()

    # else:
    #     # if not opt.longAudio:
    #     dataset = dataset.map(U.padding(opt.inputLength // 2))
    #     dataset = dataset.map(U.random_crop(opt.inputLength))
    #     dataset = dataset.map(U.normalize(float(2 ** 16 / 2)))
    #     # dataset = dataset.map(U.multi_crop(opt.inputLength, opt.nCrops))


    dataset = dataset.batch(batch_size)
    dataset = dataset.map(U.reshape([batch_size, -1, 1]))
    iterator = dataset.make_initializable_iterator()

    return iterator


def get_train_example(iterator_next):
    '''Get a training or testing sample
    '''
    if mix:  # Training phase of BC learning
        # sound1, label1, sound2, label2 = U.get_different_sounds(iterator_next)
        sound1, label1 = iterator_next
        sound2, label2 = iterator_next        
        ratio = np.array(np.random.random())
        sound = U.mix(sound1, sound2, ratio, opt.fs)
        eye = tf.eye(opt.nClasses)
        label = tf.gather(eye, label1) * ratio
        label += tf.gather(eye, label2) * (1 - ratio)

    # elif not train and opt.longAudio > 0:  # Mix two audio on long frame (for testing)
    #     sound1, label1, sound2, label2 = U.get_different_sounds(iterator)
    #     raise NotImplementedError()

    # Training phase of standard learning or testing phase
    # sound, label = iterator_next

    if opt.noiseAugment:
        sound_len = tf.shape(sound)[0]
        sound = sound + 0.01 * next_noise(is_train, sound_len)

    if train and opt.strongAugment:
        sound, label = U.random_gain(6)(sound, label)

    return sound, label


def main():
    # For testing purposes
    print('---')
    train_iterator = dataset_input_fn(False, 8)
    with tf.Session() as sess:
        import matplotlib.pyplot as plt
        # sess.run(train_iterator.initializer)
        
        for _ in range(10):
            results, = sess.run([train_iterator])
            a, lbl = results
            a = a[0,:,0]
            print(lbl[0])
            plt.plot(a)
            plt.show()


if __name__ == '__main__':
    main()

