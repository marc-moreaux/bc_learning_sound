"""
 Implementation of EnvNet-v2 (ours)
 opt.fs = 44100
 opt.inputLength = 66650

"""

import chainer
import chainer.functions as F
import chainer.links as L
from convbnrelu import ConvBNReLU


class EnvNetv2(chainer.Chain):
    def __init__(self, n_classes, **kwargs):
        super(EnvNetv2, self).__init__(
            conv1=ConvBNReLU(1, 32, (1, 64), stride=(1, 2)),
            conv2=ConvBNReLU(32, 64, (1, 16), stride=(1, 2)),
            conv3=ConvBNReLU(1, 32, (8, 8)),
            conv4=ConvBNReLU(32, 32, (8, 8)),
            conv5=ConvBNReLU(32, 64, (1, 4)),
            conv6=ConvBNReLU(64, 64, (1, 4)),
            conv7=ConvBNReLU(64, 128, (1, 2)),
            conv8=ConvBNReLU(128, 128, (1, 2)),
            conv9=ConvBNReLU(128, 256, (1, 2)),
            conv10=ConvBNReLU(256, 256, (1, 2)),
            conv11=ConvBNReLU(256, n_classes, (1, 1)),
            fc11=L.Linear(256 * 10 * 8, 4096),
            fc12=L.Linear(4096, 4096),
            fc13=L.Linear(4096, n_classes)
        )
        self.train = True
        if 'GAP' in kwargs.keys():
            self.use_GAP = kwargs['GAP']

    def __call__(self, x):
        h = self.conv1(x, self.train)
        h = self.conv2(h, self.train)
        h = F.max_pooling_2d(h, (1, 64))
        h = F.swapaxes(h, 1, 2)

        h = self.conv3(h, self.train)
        h = self.conv4(h, self.train)
        h = F.max_pooling_2d(h, (5, 3))
        h = self.conv5(h, self.train)
        h = self.conv6(h, self.train)
        h = F.max_pooling_2d(h, (1, 2))
        h = self.conv7(h, self.train)
        h = self.conv8(h, self.train)
        h = F.max_pooling_2d(h, (1, 2))
        h = self.conv9(h, self.train)
        h = self.conv10(h, self.train)
        h = F.max_pooling_2d(h, (1, 2))

        if self.use_GAP:
            h = self.conv11(h, self.train)
            self.maps = h
            batch, channels, height, width = h.data.shape
            h = F.reshape(F.average_pooling_2d(h, (height, width)), (batch, channels))
            return h

        h = F.dropout(F.relu(self.fc11(h)), train=self.train)
        h = F.dropout(F.relu(self.fc12(h)), train=self.train)

        return self.fc13(h)
