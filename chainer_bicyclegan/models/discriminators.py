import chainer
import chainer.functions as F
import chainer.links as L

from chainer_cyclegan.links import InstanceNormalization

from ..initializers import XavierNormal
from .sequential import Sequential


class D_NLayersMulti(chainer.Chain):

    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer='instance', use_sigmoid=False, num_D=2):
        super(D_NLayersMulti, self).__init__()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            with self.init_scope():
                self.model = Sequential(*layers)
        else:
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            model = [Sequential(*layers)]
            self.down = lambda x: F.average_pooling_2d(x, 3, stride=2, pad=1)
            for i in range(num_D - 1):
                ndf = int(round(ndf / (2 ** (i + 1))))
                layers = self.get_layers(
                    input_nc, ndf, n_layers, norm_layer, use_sigmoid)
                model.append(Sequential(*layers))
            with self.init_scope():
                self.model = Sequential(*model)

    def get_layers(self, input_nc, ndf=64, n_layers=3,
                   norm_layer='instance', use_sigmoid=False):
        assert norm_layer == 'instance'
        norm_layer = lambda size: InstanceNormalization(
            size, decay=0.9, eps=1e-05, use_beta=False, use_gamma=False)

        kw = 4
        padw = 1
        sequence = [
            L.Convolution2D(input_nc, ndf, ksize=kw, stride=2, pad=padw,
                            initialW=XavierNormal()),
            lambda x: F.leaky_relu(x, 0.2),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                L.Convolution2D(ndf * nf_mult_prev, ndf * nf_mult,
                                ksize=kw, stride=2, pad=padw,
                                initialW=XavierNormal()),
                norm_layer(ndf * nf_mult),
                lambda x: F.leaky_relu(x, 0.2),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            L.Convolution2D(ndf * nf_mult_prev, ndf * nf_mult,
                            ksize=kw, stride=1, pad=padw,
                            initialW=XavierNormal()),
            norm_layer(ndf * nf_mult),
            lambda x: F.leaky_relu(x, 0.2),
        ]

        sequence += [
            L.Convolution2D(ndf * nf_mult, 1, ksize=kw, stride=1, pad=padw,
                            initialW=XavierNormal()),
        ]

        if use_sigmoid:
            sequence += [lambda x: F.sigmoid(x)]
        return sequence

    def __call__(self, x):
        if self.num_D == 1:
            return self.model(x)

        result = []
        down = x
        for i in range(self.num_D):
            result.append(self.model[i](down))
            if i != self.num_D - 1:
                down = self.down(down)
        return result
