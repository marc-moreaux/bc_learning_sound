"""
 Implementation of EnvNet-v2 (ours)
 opt.fs = 44100
 opt.inputLength = 66650

"""

import chainer
import chainer.functions as F
import chainer.links as L
from .convbnrelu import ConvBNReLU


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
            fc13=L.Linear(4096, n_classes),
        )
        self.train = True
        self.use_GAP = False
        self.use_bypass = False 
        self.maps = []
        if 'GAP' in list(kwargs.keys()):
            self.use_GAP = kwargs['GAP']
        if 'bypass' in list(kwargs.keys()):
            self.use_bypass = kwargs['bypass']
            self.BPlayers = []

        if self.use_bypass:
            self.convBP0=ConvBNReLU(1, n_classes, (1, 1)),
            self.convBP1=ConvBNReLU(32, n_classes, (1, 1)),
            self.convBP2=ConvBNReLU(64, n_classes, (1, 1)),
 
    def bypass(self, conv, prev):
        maps = conv(prev, self.train)
        batch, channels, height, width = maps.data.shape
        out = F.reshape(F.average_pooling_2d(maps, (height, width)), (batch, channels))
        self.maps.append(maps)
        self.BPlayers.append(out)

    def __call__(self, x):
        self.maps = []
        self.BPlayers = []

        h = self.conv1(x, self.train)
        h = self.conv2(h, self.train)
        h = F.max_pooling_2d(h, (1, 64))
        h = F.swapaxes(h, 1, 2)

        if self.use_bypass:
            self.bypass(self.convBP0, h)
        h = self.conv3(h, self.train)
        h = self.conv4(h, self.train)
        h = F.max_pooling_2d(h, (5, 3))

        if self.use_bypass:
            self.bypass(self.convBP1, h)
        h = self.conv5(h, self.train)
        h = self.conv6(h, self.train)
        h = F.max_pooling_2d(h, (1, 2))
        
        if self.use_bypass:
            self.bypass(self.convBP2, h)
        h = self.conv7(h, self.train)
        h = self.conv8(h, self.train)
        h = F.max_pooling_2d(h, (1, 2))

        h = self.conv9(h, self.train)
        h = self.conv10(h, self.train)

        if self.use_GAP and self.use_bypass:
            self.bypass(self.conv11, h)
            BPlayers = F.stack(self.BPlayers, axis=2)
            h = F.sum(BPlayers, axis=2)
            return h

        if self.use_GAP:
            h = self.conv11(h, self.train)
            self.maps = h
            batch, channels, height, width = h.data.shape
            h = F.reshape(F.average_pooling_2d(h, (height, width)), (batch, channels))
            return h

        h = F.max_pooling_2d(h, (1, 2))
        h = F.dropout(F.relu(self.fc11(h)))
        h = F.dropout(F.relu(self.fc12(h)))

        return self.fc13(h)
