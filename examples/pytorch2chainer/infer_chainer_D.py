#!/usr/bin/env python

import argparse
import os.path as osp

import chainer
from chainer import cuda
import numpy as np
import skimage.io

import cv2  # NOQA

from chainer_bicyclegan.models import D_NLayersMulti


here = osp.dirname(osp.abspath(__file__))

chainer.global_config.train = False
chainer.global_config.enable_backprop = False

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id')
args = parser.parse_args()

gpu = args.gpu
output_nc = 3
img_file = osp.join(here, 'data/edges2shoes_val_100_AB.jpg')
D_model_file = osp.join(here, 'data/edges2shoes_net_D_from_chainer.npz')

print('GPU id: %d' % gpu)
print('D model: %s' % D_model_file)
print('Input file: %s' % img_file)

D = D_NLayersMulti(
    input_nc=output_nc,
    ndf=64,
    n_layers=3,
    norm_layer='instance',
    use_sigmoid=False,
    num_D=2,
)
chainer.serializers.load_npz(D_model_file, D)
cuda.get_device_from_id(gpu).use()
D.to_gpu()

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

y = D(x_B)
for i, yi in enumerate(y):
    yi = cuda.to_cpu(yi.array)
    print(i, yi.shape, (yi.min(), yi.mean(), yi.max()))
