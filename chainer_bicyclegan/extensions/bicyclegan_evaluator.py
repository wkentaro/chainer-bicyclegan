import copy
import os
import os.path as osp

import chainer
from chainer import cuda
from chainer.dataset import convert
import chainer.functions as F
from chainer import training
from chainer import Variable
import fcn
import numpy as np
import skimage.io


def img_as_ubyte(img):
    return ((img + 1) / 2. * 255).astype(np.uint8)


def get_z(mu, logvar):
    std = F.exp(logvar * 0.5)
    batchsize = std.shape[0]
    nz = std.shape[1]
    eps = np.random.normal(0, 1, (batchsize, nz))
    eps = eps.astype(np.float32)
    eps = chainer.Variable(cuda.to_gpu(eps))
    return (eps * std) + mu


class BicycleGANEvaluator(training.Extension):

    trigger = (1, 'epoch')

    def __init__(self, iterator,
                 converter=convert.concat_examples,
                 device=None,
                 shape=(6, 6)):
        self._iterator = iterator
        self.converter = converter
        self.device = device
        self._shape = shape

    def __call__(self, trainer):
        E = trainer.updater.get_optimizer('E').target
        G = trainer.updater.get_optimizer('G').target

        iterator = self._iterator
        it = copy.copy(iterator)

        rows = []
        for batch in it:
            assert len(batch) == 1

            img_A, img_B = zip(*batch)
            row = [
                img_as_ubyte(img_A[0].transpose(1, 2, 0)),
                img_as_ubyte(img_B[0].transpose(1, 2, 0)),
            ]

            img_A = np.asarray(img_A)[:, 0:1, :, :]
            img_B = np.asarray(img_B)

            real_A = Variable(self.converter(img_A, self.device))
            real_B = Variable(self.converter(img_B, self.device))

            n_rows, n_cols = self._shape

            nz = 8

            np.random.seed(0)
            z_samples = np.random.normal(
                0, 1, (n_cols - 1, nz)).astype(np.float32)

            for i in range(0, n_cols):
                if i == 0:
                    with chainer.using_config('enable_backprop', False), \
                            chainer.using_config('train', False):
                        mu, logvar = E(real_B)
                    z = get_z(mu, logvar)
                else:
                    z = cuda.to_gpu(z_samples[i - 1][None])
                    z = chainer.Variable(z)

                with chainer.using_config('enable_backprop', False), \
                        chainer.using_config('train', False):
                    y = G(real_A, z)

                fake_B = cuda.to_cpu(y.array[0].transpose(1, 2, 0))
                fake_B = img_as_ubyte(fake_B)
                row.append(fake_B)

            row = fcn.utils.get_tile_image(row, tile_shape=(1, len(row)))
            rows.append(row)

            if len(rows) >= n_rows:
                break

        out_file = osp.join(
            trainer.out, 'evaluations', '%08d.jpg' % trainer.updater.epoch)
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        out = fcn.utils.get_tile_image(rows, tile_shape=(len(rows), 1))
        skimage.io.imsave(out_file, out)
