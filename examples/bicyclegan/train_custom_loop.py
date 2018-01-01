#!/usr/bin/env python

from __future__ import print_function

import argparse
import datetime
import os
import os.path as osp
import sys
import time

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.optimizers as O
import chainer.serializers as S
import chainercv
# import cupy as cp
import numpy as np
import PIL.Image
import skimage.io

from chainer_cyclegan.datasets import BerkeleyPix2PixDataset

here = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(here, '../pytorch2chainer'))

from lib.models import D_NLayersMulti
from lib.models import E_ResNet
from lib.models import G_Unet_add_all


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
        if self._train:
            img_A, param_crop = chainercv.transforms.random_crop(
                img_A, size=self._fine_size, return_param=True)
            # img_A, param_flip = chainercv.transforms.random_flip(
            #     img_A, x_random=True, return_param=True)
        img_A = img_A.astype(np.float32) / 255  # ToTensor
        img_A = (img_A - 0.5) / 0.5  # Normalize

        img_B = img_B.transpose(2, 0, 1)
        img_B = chainercv.transforms.resize(
            img_B, size=self._load_size,
            interpolation=PIL.Image.BICUBIC)
        if self._train:
            img_B = img_B[:, param_crop['y_slice'], param_crop['x_slice']]
            # img_B = chainercv.transforms.flip(
            #     img_B, x_flip=param_flip['x_flip'])
        img_B = img_B.astype(np.float32) / 255  # ToTensor
        img_B = (img_B - 0.5) / 0.5  # Normalize

        return img_A, img_B


def backward_D(D, real, fake):
    xp = cuda.get_array_module(real.array)

    # Real
    pred_real = D(real)
    loss_D_real = 0
    for pr in pred_real:
        loss_D_real += F.mean_squared_error(pr, xp.ones_like(pr.array))

    # Fake, stop backprop to the generator by detaching fake_B
    pred_fake = D(fake.array)
    loss_D_fake = 0
    for pf in pred_fake:
        loss_D_fake += F.mean_squared_error(pf, xp.zeros_like(pf.array))

    # Combined loss
    loss_D = loss_D_fake + loss_D_real
    loss_D.backward()
    return loss_D, [loss_D_fake, loss_D_real]


def backward_G_GAN(fake, D, ll):
    xp = cuda.get_array_module(fake.array)

    if ll > 0.0:
        with chainer.using_config('enable_backprop', False):
            pred_fake = D(fake)
        loss_G_GAN = 0
        for pf in pred_fake:
            loss_G_GAN += F.mean_squared_error(pf, xp.ones_like(pf.array))
    else:
        loss_G_GAN = 0
    return loss_G_GAN * ll


def backward_EG(fake_data_encoded, fake_data_random,
                fake_B_encoded, real_B_encoded,
                D, D2, lambda_GAN, lambda_GAN2,
                mu, logvar):
    lambda_kl = 0.01
    lambda_L1 = 10.0

    # 1, G(A) should fool D
    loss_G_GAN = backward_G_GAN(fake_data_encoded, D, lambda_GAN)
    loss_G_GAN2 = backward_G_GAN(fake_data_random, D2, lambda_GAN2)
    # 2. KL loss
    if lambda_kl > 0:
        kl_element = (((mu * mu) + F.exp(logvar)) * -1) + 1 + logvar
        loss_kl = F.sum(kl_element) * -0.5 * lambda_kl
    else:
        loss_kl = 0
    # 3, reconstruction |fake_B-real_B|
    if lambda_L1 > 0:
        loss_G_L1 = F.mean_absolute_error(fake_B_encoded, real_B_encoded)
        loss_G_L1 = lambda_L1 * loss_G_L1
    else:
        loss_G_L1 = 0

    loss_G = loss_G_GAN + loss_G_GAN2 + loss_G_L1 + loss_kl
    loss_G.backward()
    return loss_G, loss_G_GAN, loss_G_GAN2, loss_G_L1, loss_kl


def backward_G_alone(lambda_z, mu2, z_random):
    # 3, reconstruction |z_predit-z_random|
    if lambda_z > 0.0:
        loss_z_L1 = F.mean(F.absolute(mu2 - z_random)) * lambda_z
        loss_z_L1.backward()
    else:
        loss_z_L1 = 0
    return loss_z_L1


def get_z_random(size0, size1):
    z_random = np.random.normal(0, 1, (size0, size1))
    z_random = z_random.astype(np.float32)
    z_random = cuda.to_gpu(z_random)
    return chainer.Variable(z_random)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True)
    args = parser.parse_args()

    gpu = args.gpu

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
        cuda.get_device_from_id(gpu).use()
        E.to_gpu()
        G.to_gpu()
        D.to_gpu()
        D2.to_gpu()

    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999

    optimizer_E = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
    optimizer_E.setup(E)

    optimizer_G = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
    optimizer_G.setup(G)

    optimizer_D = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
    optimizer_D.setup(D)

    optimizer_D2 = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
    optimizer_D2.setup(D2)

    batch_size = 2
    dataset = BerkeleyPix2PixDataset('edges2shoes', split='train')
    dataset = chainer.datasets.TransformDataset(dataset, BicycleGANTransform())
    iterator = chainer.iterators.SerialIterator(dataset, batch_size=batch_size)

    epoch_count = 1
    niter = 100
    niter_decay = 100

    def lambda_rule(epoch):
        lr_l = 1.0 - (max(0, epoch + 1 + epoch_count - niter) /
                      float(niter_decay + 1))
        return lr_l

    out_dir = osp.join(
        'logs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    with open(osp.join(out_dir, 'log.csv'), 'w') as f:
        f.write(','.join([
            'epoch',
            'iteration',
            'loss_D',
            'loss_D2',
            'loss_G',
            'loss_G_GAN',
            'loss_G_GAN2',
            'loss_G_L1',
            'loss_kl',
            'loss_z_L1',
        ]))
        f.write('\n')

    max_epoch = niter + niter_decay - epoch_count
    dataset_size = len(dataset)
    for epoch in range(epoch_count, niter + niter_decay + 1):
        t_start = time.time()

        for iteration in range(dataset_size // batch_size):
            batch = next(iterator)
            if len(batch) != batch_size:
                continue

            img_A, img_B = zip(*batch)
            img_A = np.asarray(img_A)[:, 0:1, :, :]
            img_B = np.asarray(img_B)

            assert batch_size == 2
            assert len(img_A) == 2
            assert len(img_B) == 2
            real_A_encoded = img_A[0:1]
            real_A_random = img_A[1:2]
            real_B_encoded = img_B[0:1]
            real_B_random = img_B[1:2]
            if gpu >= 0:
                real_A_encoded = cuda.to_gpu(real_A_encoded)
                real_A_random = cuda.to_gpu(real_A_random)
                real_B_encoded = cuda.to_gpu(real_B_encoded)
                real_B_random = cuda.to_gpu(real_B_random)
            real_A_encoded = chainer.Variable(real_A_encoded)
            real_A_random = chainer.Variable(real_A_random)
            real_B_encoded = chainer.Variable(real_B_encoded)
            real_B_random = chainer.Variable(real_B_random)

            # update D
            # -----------------------------------------------------------------
            # forward {{
            mu, logvar = E(real_B_encoded)
            std = F.exp(logvar * 0.5)

            eps = get_z_random(std.shape[0], std.shape[1])
            z_encoded = (eps * std) + mu

            z_random = get_z_random(real_A_random.shape[0], std.shape[1])

            fake_B_encoded = G(real_A_encoded, z_encoded)

            # generate fake_B_random
            fake_B_random = G(real_A_encoded, z_random)

            fake_data_encoded = fake_B_encoded
            fake_data_random = fake_B_random
            real_data_encoded = real_B_encoded
            real_data_random = real_B_random

            lambda_z = 0.5

            mu2, logvar2 = E(fake_B_random)
            # std2 = F.exp(logvar2 * 0.5)
            # eps2 = get_z_random(std2.shape[0], std2.shape[1])
            # z_predict = (eps2 * std2) + mu2

            # }} forward

            # update D1
            lambda_GAN = 1.0
            lambda_GAN2 = 1.0
            if lambda_GAN > 0:
                D.cleargrads()
                loss_D, losses_D = backward_D(
                    D, real_data_encoded, fake_data_encoded)
                optimizer_D.update()

            # update D2
            if lambda_GAN2 > 0:
                D2.cleargrads()
                loss_D2, losses_D2 = backward_D(
                    D2, real_data_random, fake_data_random)
                optimizer_D2.update()

            # update G
            # -----------------------------------------------------------------
            E.cleargrads()
            G.cleargrads()
            loss_G, loss_G_GAN, loss_G_GAN2, loss_G_L1, loss_kl = backward_EG(
                fake_data_encoded, fake_data_random,
                fake_B_encoded, real_B_encoded,
                D, D2, lambda_GAN, lambda_GAN2,
                mu, logvar)
            optimizer_G.update()
            optimizer_E.update()

            # update G only
            if lambda_z > 0.0:
                G.cleargrads()
                E.cleargrads()
                loss_z_L1 = backward_G_alone(lambda_z, mu2, z_random)
                optimizer_G.update()

            # log
            # -----------------------------------------------------------------
            if iteration % (100 // batch_size) == 0:
                time_per_iter1 = ((time.time() - t_start) /
                                  (iteration + 1) / batch_size)

                if hasattr(loss_D, 'array'):
                    loss_D = float(loss_D.array)
                if hasattr(loss_D2, 'array'):
                    loss_D2 = float(loss_D2.array)
                if hasattr(loss_G, 'array'):
                    loss_G = float(loss_G.array)
                if hasattr(loss_G_GAN, 'array'):
                    loss_G_GAN = float(loss_G_GAN.array)
                if hasattr(loss_G_GAN2, 'array'):
                    loss_G_GAN2 = float(loss_G_GAN2.array)
                if hasattr(loss_G_L1, 'array'):
                    loss_G_L1 = float(loss_G_L1.array)
                if hasattr(loss_kl, 'array'):
                    loss_kl = float(loss_kl.array)
                if hasattr(loss_z_L1, 'array'):
                    loss_z_L1 = float(loss_z_L1.array)

                print('-' * 79)
                print(
                    'Epoch: {:d}/{:d} ({:.1%}), '
                    'Iteration: {:d}/{:d} ({:.1%}), Time: {:f}'
                    .format(epoch, max_epoch, 1. * epoch / max_epoch,
                            batch_size * iteration, dataset_size,
                            1. * batch_size * iteration / dataset_size,
                            time_per_iter1))

                print('D: {:.2f}'.format(loss_D),
                      'D2: {:.2f}'.format(loss_D2),
                      'G: {:.2f}'.format(loss_G),
                      'G_GAN: {:.2f}'.format(loss_G_GAN),
                      'G_GAN2: {:.2f}'.format(loss_G_GAN2),
                      'G_L1: {:.2f}'.format(loss_G_L1),
                      'kl: {:.2f}'.format(loss_kl),
                      'z_L1: {:.2f}'.format(loss_z_L1))

                with open(osp.join(out_dir, 'log.csv'), 'a') as f:
                    f.write(','.join(map(str, [
                        epoch,
                        ((epoch - 1) * dataset_size) + iteration * batch_size,
                        loss_D,
                        loss_D2,
                        loss_G,
                        loss_G_GAN,
                        loss_G_GAN2,
                        loss_G_L1,
                        loss_kl,
                        loss_z_L1,
                    ])))
                    f.write('\n')

        # visualize
        # -------------------------------------------------------------------------
        real_A_encoded = real_A_encoded.array[0].transpose(1, 2, 0)
        real_A_encoded = np.repeat(real_A_encoded, 3, axis=2)
        real_A_encoded = cuda.to_cpu(real_A_encoded)
        real_B_encoded = real_B_encoded.array[0].transpose(1, 2, 0)
        real_B_encoded = cuda.to_cpu(real_B_encoded)
        real_A_random = real_A_random.array[0].transpose(1, 2, 0)
        real_A_random = np.repeat(real_A_random, 3, axis=2)
        real_A_random = cuda.to_cpu(real_A_random)
        real_B_random = real_B_random.array[0].transpose(1, 2, 0)
        real_B_random = cuda.to_cpu(real_B_random)
        fake_B_encoded = fake_B_encoded.array[0].transpose(1, 2, 0)
        fake_B_encoded = cuda.to_cpu(fake_B_encoded)
        fake_B_random = fake_B_random.array[0].transpose(1, 2, 0)
        fake_B_random = cuda.to_cpu(fake_B_random)
        viz = np.vstack([np.hstack([real_A_encoded, real_B_encoded]),
                         np.hstack([real_A_random, real_B_random]),
                         np.hstack([fake_B_encoded, fake_B_random])])
        skimage.io.imsave(osp.join(out_dir, '{:08}.jpg'.format(epoch)), viz)

        S.save_npz(osp.join(out_dir, '{:08}_E.npz'.format(epoch)), E)
        S.save_npz(osp.join(out_dir, '{:08}_G.npz'.format(epoch)), G)
        S.save_npz(osp.join(out_dir, '{:08}_D.npz'.format(epoch)), D)
        S.save_npz(osp.join(out_dir, '{:08}_D2.npz'.format(epoch)), D2)

        # update learning rate
        # -------------------------------------------------------------------------
        lr_new = lambda_rule(epoch)
        optimizer_E.alpha *= lr_new
        optimizer_G.alpha *= lr_new
        optimizer_D.alpha *= lr_new
        optimizer_D2.alpha *= lr_new


if __name__ == '__main__':
    main()
