import chainer
import chainer.functions as F
import chainer.links as L


class ConvBNReLU(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 initialW=chainer.initializers.HeNormal(), nobias=True, use_bn=True):
        super(ConvBNReLU, self).__init__(
            conv=L.Convolution2D(in_channels, out_channels, ksize, stride, pad,
                                 initialW=initialW, nobias=nobias),
            bn=L.BatchNormalization(out_channels)
        )
        self.use_bn=use_bn

    def __call__(self, x, train):
        h = self.conv(x)

        if self.use_bn:
            h = self.bn(h)

        return F.relu(h)


class ConvBNSig(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 initialW=chainer.initializers.HeNormal(), nobias=True, use_bn=True):
        super(ConvBNSig, self).__init__(
            conv=L.Convolution2D(in_channels, out_channels, ksize, stride, pad,
                                 initialW=initialW, nobias=nobias),
            bn=L.BatchNormalization(out_channels)
        )
        self.use_bn=use_bn

    def __call__(self, x, train):
        h = self.conv(x)
        if self.use_bn:
            h = self.bn(h)

        return F.sigmoid(h)


class ConvBN(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 initialW=chainer.initializers.HeNormal(), nobias=True, use_bn=True):
        super(ConvBN, self).__init__(
            conv=L.Convolution2D(in_channels, out_channels, ksize, stride, pad,
                                 initialW=initialW, nobias=nobias),
            bn=L.BatchNormalization(out_channels)
        )
        self.use_bn=use_bn

    def __call__(self, x, train):
        h = self.conv(x)
        if self.use_bn:
            h = self.bn(h)

        return h
