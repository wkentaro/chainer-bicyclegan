#!/usr/bin/env python

import argparse
import os
import os.path as osp
import sys

import numpy as np
import skimage.io
import torch

import cv2  # NOQA

here = osp.dirname(osp.abspath(__file__))
pytorch_dir = osp.realpath(osp.join(here, '../../src/pytorch-bicyclegan'))
sys.path.insert(0, pytorch_dir)

from models.networks import D_NLayersMulti
from models.networks import get_norm_layer


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id')
args = parser.parse_args()

gpu = args.gpu
output_nc = 3
img_file = osp.join(here, 'data/edges2shoes_val_100_AB.jpg')
D_model_file = osp.join(here, 'data/edges2shoes_net_D.pth')

print('GPU id: %d' % gpu)
print('D model: %s' % D_model_file)
print('Input file: %s' % img_file)

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

D = D_NLayersMulti(
    input_nc=output_nc,
    ndf=64,
    n_layers=3,
    norm_layer=get_norm_layer('instance'),
    use_sigmoid=False,
    gpu_ids=[],
    num_D=2,
)
D.load_state_dict(torch.load(D_model_file))
D.cuda()

img = skimage.io.imread(img_file)
H, W = img.shape[:2]
real_A = img[:, :W // 2, :]
real_A = real_A[:, :, 0:1]   # edges
real_B = img[:, W // 2:, :]  # shoes

xi_A = real_A.astype(np.float32) / 255. * 2 - 1
x_A = xi_A.transpose(2, 0, 1)[None]
x_A = torch.from_numpy(x_A).cuda()
x_A = torch.autograd.Variable(x_A, volatile=True)

xi_B = real_B.astype(np.float32) / 255. * 2 - 1
x_B = xi_B.transpose(2, 0, 1)[None]
x_B = torch.from_numpy(x_B).cuda()
x_B = torch.autograd.Variable(x_B, volatile=True)

y = D(x_B)
for i, yi in enumerate(y):
    yi = yi.data.cpu().numpy()
    print(i, yi.shape, (yi.min(), yi.mean(), yi.max()))
