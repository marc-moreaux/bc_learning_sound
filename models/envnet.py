"""
 Implementation of EnvNet [Tokozume and Harada, 2017]
 opt.fs = 16000
 opt.inputLength = 24014

"""

import chainer
import chainer.functions as F
import chainer.links as L
from .convbnrelu import ConvBNReLU, ConvBNSig, ConvBN


_opt = {'stride': 'same',
        'use_attention': -1}


def _max_pooling_2d(x, ksize, _opt):
    if _opt['stride'] == 'same':
        return F.max_pooling_2d(x, ksize)
    else:
        stride = list(ksize)
        stride[-1] = 1
        return F.max_pooling_2d(x, ksize, stride)


class EnvNet(chainer.Chain):
    def __init__(self, n_classes, **kwargs):
        super(EnvNet, self).__init__(
            conv1=ConvBNReLU(1, 40, (1, 8)),
            conv2=ConvBNReLU(40, 40, (1, 8)),
            conv3=ConvBNReLU(1, 50, (8, 13)),
            conv4=ConvBNReLU(50, 50, (1, 5)),
            fc5=L.Linear(50 * 11 * 14, 4096),
            fc6=L.Linear(4096, 4096),
            fc7=L.Linear(4096, n_classes),

            conv5=ConvBNReLU(50, 50, (3, 3), use_bn=False),
            convAtt4 = ConvBNSig(50, 50, (1, 5)),
            convAtt5 = ConvBNReLU(50, 50, (1, 5)),
            fcAtt5=L.Linear(50 * 11 * 26, 1024),
            fcAtt6=L.Linear(1024, 3),
            convGAP=ConvBNReLU(50, n_classes, (1, 1)),
        )
        self.train = True
        if _opt['use_attention'] == 1:
            self.convAtt4 = ConvBNSig(50, 50, (1, 5), pad=(0,1))
        if _opt['use_attention'] == 2:
            self.convAtt4 = ConvBNReLU(50, 50, (1, 5))
        if _opt['use_attention'] == 3:
            self.convAtt4 = ConvBNReLU(50, 50, (1, 5), pad=(0,1))
        if _opt['use_attention'] == 4:
            self.convAtt4 = ConvBNSig(50, 50, (1, 5), pad=(0,1))
        if _opt['use_attention'] == 5:
            self.convAtt4 = ConvBN(50, 50, (1, 5), pad=(0,1))
        if _opt['use_attention'] == 6:
            self.convAtt4 = ConvBN(50, 50, (1, 5), pad=(0,2))
        if _opt['use_attention'] == 7:
            self.convAtt4 = ConvBNReLU(50, 50, (1, 5), pad=(0,2))
        
        self.use_GAP = False
        if 'GAP' in list(kwargs.keys()):
            self.use_GAP = kwargs['GAP']

    def __call__(self, x):
        h = self.conv1(x, self.train)
        h = self.conv2(h, self.train)
        h = F.max_pooling_2d(h, (1, 160))
        h = F.swapaxes(h, 1, 2)

        h = self.conv3(h, self.train)
        h3 = _max_pooling_2d(h, 3, _opt)
        self.h3 = h3

        h = self.conv4(h3, self.train)
        self.h = h

        # Attention mechanism
        if _opt['use_attention'] == 0:
            h_att = self.convAtt4(h3, self.train)
            h = h * h_att

        if _opt['use_attention'] == 1:
            h_att = F.max_pooling_2d(h3, 2)
            h_att = self.convAtt4(h_att, self.train)
            h_att = F.unpooling_2d(h_att, 2, outsize=h.shape[-2:])
            h = h * h_att

        if _opt['use_attention'] == 2:
            h_att = self.convAtt4(h3, self.train)
            h_att = F.softmax(h_att, axis=3)
            h = h * h_att

        if _opt['use_attention'] == 3:
            h_att = F.max_pooling_2d(h3, 2)
            h_att = self.convAtt4(h_att, self.train)
            h_att = F.softmax(h_att, axis=3)
            h_att = F.unpooling_2d(h_att, 2, outsize=h.shape[-2:])
            h = h * h_att
        
        if _opt['use_attention'] == 4:
            h_att = F.max_pooling_2d(h3, 2)
            h_att = self.convAtt4(h_att, self.train)
            h_att = F.softmax(h_att, axis=3)
            h_att = F.unpooling_2d(h_att, 2, outsize=h.shape[-2:])
            h = h * h_att
        
        if _opt['use_attention'] == 5:
            h_att = F.max_pooling_2d(h3, 2)
            h_att = self.convAtt4(h_att, self.train)
            h_att = F.softmax(h_att, axis=3)
            h_att = F.unpooling_2d(h_att, 2, outsize=h.shape[-2:])
            h = h * h_att
        
        if _opt['use_attention'] == 6:
            h_att = F.max_pooling_2d(h3, 5)
            h_att = self.convAtt4(h_att, self.train)
            h_att = F.softmax(h_att, axis=3)
            h_att = F.unpooling_2d(h_att, 5, outsize=h.shape[-2:])
            h = h * h_att

        if _opt['use_attention'] == 7:
            h_att = F.max_pooling_2d(h3, (1, 5))
            h_att = self.convAtt4(h_att, self.train)
            h_att = F.softmax(h_att, axis=3)
            h_att = F.unpooling_2d(h_att, (1,5), outsize=self.h.shape[-2:])
            h = h * h_att

        h = _max_pooling_2d(h, (1, 3), _opt)

        if self.use_GAP:
            h = self.conv5(h, self.train)
            h = self.convGAP(h, self.train)
            self.maps = h
            batch, channels, height, width = h.data.shape
            h = F.reshape(F.average_pooling_2d(h, (height, width)), (batch, channels))
            return h

        try:
            h = F.dropout(F.relu(self.fc5(h)), train=self.train)
            h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        except:
            print(h.shape)

        return self.fc7(h)

"""
 Implementation of EnvNet [Tokozume and Harada, 2017]
 opt.fs = 16000
 opt.inputLength = 24014
"""

import chainer
import chainer.functions as F
import chainer.links as L


class EnvNet(chainer.Chain):
    def __init__(self, n_classes, **kwargs):
        super(EnvNet, self).__init__(
            conv1=ConvBNReLU(1, 40, (1, 8)),
            conv2=ConvBNReLU(40, 40, (1, 8)),
            conv3=ConvBNReLU(1, 50, (8, 13)),
            conv4=ConvBNReLU(50, 50, (1, 5)),
            fc5=L.Linear(50 * 11 * 14, 4096),
            fc6=L.Linear(4096, 4096),
            fc7=L.Linear(4096, n_classes)
        )
        self.train = True

    def __call__(self, x):
        h = self.conv1(x, self.train)
        h = self.conv2(h, self.train)
        h = F.max_pooling_2d(h, (1, 160))
        h = F.swapaxes(h, 1, 2)

        h = self.conv3(h, self.train)
        h = F.max_pooling_2d(h, 3)
        h = self.conv4(h, self.train)
        h = F.max_pooling_2d(h, (1, 3))

        h = F.dropout(F.relu(self.fc5(h)))
        h = F.dropout(F.relu(self.fc6(h)))

        return self.fc7(h)
