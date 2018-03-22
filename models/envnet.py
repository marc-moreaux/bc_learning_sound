"""
 Implementation of EnvNet [Tokozume and Harada, 2017]
 opt.fs = 16000
 opt.inputLength = 24014

"""

import chainer
import chainer.functions as F
import chainer.links as L
from convbnrelu import ConvBNReLU


_opt = {'stride': 'same',
        'use_attention': True}


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
            fc5=L.Linear(50 * 11 * 26, 4096),
            fc6=L.Linear(4096, 4096),
            fc7=L.Linear(4096, n_classes),

            conv5=ConvBNReLU(50, 50, (3, 3)),
            convAtt4=ConvBNReLU(50, 50, (1, 5)),
            fcAtt5=L.Linear(50 * 11 * 26, 1024),
            fcAtt6=L.Linear(1024, 3),
            convGAP=ConvBNReLU(50, n_classes, (1, 1)),
        )
        self.train = True
        if 'GAP' in kwargs.keys():
            self.use_GAP = kwargs['GAP']

    def __call__(self, x):
        h = self.conv1(x, self.train)
        h = self.conv2(h, self.train)
        h = F.max_pooling_2d(h, (1, 160))
        h = F.swapaxes(h, 1, 2)

        h = self.conv3(h, self.train)
        h3 = _max_pooling_2d(h, 3, _opt)

        h = self.conv4(h3, self.train)

        # Attention mechanism
        if _opt['use_attention']:
            h_att = self.convAtt4(h3, self.train)
            
            h_att = F.sigmoid(h_att)
            h = h * h_att

            # # Other way...
            # h_att = self.convAtt4(h3, self.train)
            # h_att = F.max_pooling_2d(h, 3)
            # h_att = F.dropout(F.relu(fcAtt5(h)), train=train)
            # h_att = fcAtt6(h)

            # n_padding = x_len / 2
            # center, width, gain = F.split_axis(h, 3, axis=1)  # split resulting vec in 2
            # center = F.tanh(center)
            # width = F.sigmoid(width)
            # gain = F.sigmoid(gain)
            # center = (center + 1) * (x_len / 2)
            # width = width * x_len
            # left = padding + center - width/2
            # right = padding + center + width/2

            # x = F.pad(x, n_padding)
            # x = x[:, 1, 1, left: right] * gain

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
            print h.shape

        return self.fc7(h)
        