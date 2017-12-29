#!/usr/bin/env python

import argparse
import os.path as osp

import chainer
from chainer import cuda
import chainer.functions as F
import cv2
import fcn
import numpy as np
import skimage.io

import lib


chainer.global_config.train = False
chainer.global_config.enable_backprop = False

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id')
args = parser.parse_args()

here = osp.dirname(osp.abspath(__file__))
img_file = osp.join(here, 'data/edges2shoes_val_100_AB.jpg')
G_model_file = osp.join(here, 'data/edges2shoes_net_G_from_chainer.npz')
E_model_file = osp.join(here, 'data/edges2shoes_net_E_from_chainer.npz')

gpu = args.gpu
nz = 8
output_nc = 3

print('GPU id: %d' % args.gpu)
print('G model: %s' % G_model_file)
print('E model: %s' % E_model_file)
print('Input file: %s' % img_file)

assert gpu >= 0
cuda.get_device_from_id(gpu).use()

G = lib.models.G_Unet_add_all(
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
chainer.serializers.load_npz(G_model_file, G)
G.to_gpu()

E = lib.models.E_ResNet(
    input_nc=output_nc,
    output_nc=nz,
    ndf=64,
    n_blocks=5,
    norm_layer='instance',
    nl_layer='lrelu',
    vaeLike=True,
)
chainer.serializers.load_npz(E_model_file, E)
E.to_gpu()

img = skimage.io.imread(img_file)
H, W = img.shape[:2]
real_A = img[:, :W // 2, :]
real_A = real_A[:, :, 0:1]   # edges
real_B = img[:, W // 2:, :]  # shoes

xi_A = real_A.astype(np.float32) / 255. * 2 - 1
x_A = xi_A.transpose(2, 0, 1)[None]
x_A = cuda.to_gpu(x_A)
x_A = chainer.Variable(x_A)

xi_B = real_B.astype(np.float32) / 255. * 2 - 1
x_B = xi_B.transpose(2, 0, 1)[None]
x_B = cuda.to_gpu(x_B)
x_B = chainer.Variable(x_B)

n_samples = 33
random_state = np.random.RandomState(0)
z_samples = random_state.normal(0, 1, (n_samples, nz)).astype(np.float32)

real_A = np.repeat(real_A, 3, axis=2)
viz = [real_A, real_B]
for i in range(1 + n_samples):
    if i == 0:
        def get_z(mu, logvar):
            std = F.exp(logvar * 0.5)
            batchsize = std.shape[0]
            nz = std.shape[1]
            eps = np.random.normal(0, 1, (batchsize, nz)).astype(np.float32)
            eps = chainer.Variable(cuda.to_gpu(eps))
            return eps * std + mu

        mu, logvar = E(x_B)
        z = get_z(mu, logvar)
    else:
        z = cuda.to_gpu(z_samples[i - 1][None])
        z = chainer.Variable(z)

    y = G(x_A, z)

    fake_B = cuda.to_cpu(y.array[0].transpose(1, 2, 0))
    fake_B = ((fake_B + 1) / 2. * 255.).astype(np.uint8)

    viz.append(fake_B)
viz = fcn.utils.get_tile_image(viz)

out_file = osp.join(here, 'logs/infer_chainer.png')
cv2.imwrite(out_file, viz[:, :, ::-1])
print('Saved file: %s' % out_file)
