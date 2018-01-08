#!/usr/bin/env python

from __future__ import print_function

import argparse
import datetime
import os
import os.path as osp
import sys

os.environ['MPLBACKEND'] = 'Agg'

import chainer
from chainer.datasets import TransformDataset
import chainer.optimizers as O
from chainer import training
from chainer.training import extensions
import chainercv
import cupy as cp
import numpy as np
import PIL.Image

from chainer_cyclegan.datasets import BerkeleyPix2PixDataset

here = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(here, '../pytorch2chainer'))

from lib.models import D_NLayersMulti
from lib.models import E_ResNet
from lib.models import G_Unet_add_all

from bicyclegan_evaluator import BicycleGANEvaluator
from updater import BicycleGANUpdater


class BicycleGANTransform(object):

    def __init__(self, train=True, load_size=(286, 286), fine_size=(256, 256)):
        self._train = train
        self._load_size = load_size
        self._fine_size = fine_size

    def __call__(self, in_data):
        img_A, img_B = in_data

        img_A = img_A.transpose(2, 0, 1)
        img_A = chainercv.transforms.resize(
            img_A, size=self._load_size,
            interpolation=PIL.Image.BICUBIC)
        img_A, param_crop = chainercv.transforms.random_crop(
            img_A, size=self._fine_size, return_param=True)
        if self._train:
            # img_A, param_flip = chainercv.transforms.random_flip(
            #     img_A, x_random=True, return_param=True)
            pass
        img_A = img_A.astype(np.float32) / 255  # ToTensor
        img_A = (img_A - 0.5) / 0.5  # Normalize

        img_B = img_B.transpose(2, 0, 1)
        img_B = chainercv.transforms.resize(
            img_B, size=self._load_size,
            interpolation=PIL.Image.BICUBIC)
        img_B = img_B[:, param_crop['y_slice'], param_crop['x_slice']]
        if self._train:
            # img_B = chainercv.transforms.flip(
            #     img_B, x_flip=param_flip['x_flip'])
            pass
        img_B = img_B.astype(np.float32) / 255  # ToTensor
        img_B = (img_B - 0.5) / 0.5  # Normalize

        return img_A, img_B


def train(dataset_train, dataset_test, gpu, suffix=''):
    batch_size = 2

    np.random.seed(0)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        cp.random.seed(0)

    # Model

    output_nc = 3
    nz = 8
    E = E_ResNet(
        input_nc=output_nc,
        output_nc=nz,
        ndf=64,
        n_blocks=5,
        norm_layer='instance',
        nl_layer='lrelu',
        vaeLike=True,
    )
    G = G_Unet_add_all(
        input_nc=1,
        output_nc=output_nc,
        nz=nz,
        num_downs=8,
        ngf=64,
        norm_layer='instance',
        nl_layer='relu',
        use_dropout=False,
        upsample='basic',
    )
    D = D_NLayersMulti(
        input_nc=output_nc,
        ndf=64,
        n_layers=3,
        norm_layer='instance',
        use_sigmoid=False,
        num_D=2,
    )
    D2 = D_NLayersMulti(
        input_nc=output_nc,
        ndf=64,
        n_layers=3,
        norm_layer='instance',
        use_sigmoid=False,
        num_D=2,
    )

    if gpu >= 0:
        E.to_gpu()
        G.to_gpu()
        D.to_gpu()
        D2.to_gpu()

    # Optimizer

    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999

    optimizer_E = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
    optimizer_G = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
    optimizer_D = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
    optimizer_D2 = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)

    optimizer_E.setup(E)
    optimizer_G.setup(G)
    optimizer_D.setup(D)
    optimizer_D2.setup(D2)

    # Dataset

    iter_train = chainer.iterators.SerialIterator(
        dataset_train, batch_size=batch_size)
    iter_test = chainer.iterators.SerialIterator(
        dataset_test, batch_size=batch_size, repeat=False, shuffle=False)

    # Updater

    epoch_count = 1
    niter = 30
    niter_decay = 30

    updater = BicycleGANUpdater(
        iterator=iter_train,
        optimizer=dict(
            E=optimizer_E,
            G=optimizer_G,
            D=optimizer_D,
            D2=optimizer_D2,
        ),
        device=gpu,
    )

    # Trainer

    out = osp.join('logs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    out += suffix
    trainer = training.Trainer(
        updater, (niter + niter_decay, 'epoch'), out=out)

    trainer.extend(extensions.snapshot_object(
        target=E, filename='E_{.updater.epoch:08}.npz'),
        trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        target=G, filename='G_{.updater.epoch:08}.npz'),
        trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        target=D, filename='D_{.updater.epoch:08}.npz'),
        trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        target=D2, filename='D2_{.updater.epoch:08}.npz'),
        trigger=(1, 'epoch'))

    interval_print = 40 // batch_size

    trainer.extend(
        extensions.LogReport(trigger=(interval_print, 'iteration')))

    assert extensions.PlotReport.available()
    trainer.extend(extensions.PlotReport(
        y_keys=['loss_D'],
        x_key='iteration', file_name='loss_D.png',
        trigger=(100 // batch_size, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=['loss_G', 'loss_G_GAN', 'loss_G_GAN2', 'loss_G_L1', 'loss_kl'],
        x_key='iteration', file_name='loss_G.png',
        trigger=(100 // batch_size, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=['loss_z_L1'],
        x_key='iteration', file_name='loss_z_L1.png',
        trigger=(100 // batch_size, 'iteration')))

    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'elapsed_time',
        'loss_D', 'loss_G',
        'loss_G_GAN', 'loss_G_GAN2',
        'loss_G_L1', 'loss_kl',
        'loss_z_L1',
    ]))

    trainer.extend(
        extensions.ProgressBar(update_interval=interval_print))

    trainer.extend(BicycleGANEvaluator(iter_test, device=gpu))

    @training.make_extension(trigger=(1, 'epoch'))
    def tune_learning_rate(trainer):
        epoch = trainer.updater.epoch

        lr_rate = 1.0 - (max(0, epoch + 1 + epoch_count - niter) /
                         float(niter_decay + 1))

        trainer.updater.get_optimizer('E').alpha *= lr_rate
        trainer.updater.get_optimizer('G').alpha *= lr_rate
        trainer.updater.get_optimizer('D').alpha *= lr_rate
        trainer.updater.get_optimizer('D2').alpha *= lr_rate

    trainer.extend(tune_learning_rate)

    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True,
                        help='GPU id.')
    args = parser.parse_args()

    dataset_train = TransformDataset(
        BerkeleyPix2PixDataset('edges2shoes', 'train'),
        BicycleGANTransform())
    dataset_test = TransformDataset(
        BerkeleyPix2PixDataset('edges2shoes', 'val'),
        BicycleGANTransform(train=False))
    train(dataset_train, dataset_test, args.gpu, suffix='_edges2shoes')
