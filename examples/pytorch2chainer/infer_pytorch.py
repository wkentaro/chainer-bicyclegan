#!/usr/bin/env python

import argparse
import os
import os.path as osp
import sys

import fcn
import numpy as np
import skimage.io
import torch

torch.backends.cudnn.benchmark = True

import cv2  # NOQA

here = osp.dirname(osp.abspath(__file__))
pytorch_dir = osp.realpath(osp.join(here, '../../src/pytorch-bicyclegan'))
sys.path.insert(0, pytorch_dir)

from models.networks import E_ResNet
from models.networks import G_Unet_add_all
from models.networks import get_non_linearity
from models.networks import get_norm_layer


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id')
args = parser.parse_args()

img_file = osp.join(here, 'data/edges2shoes_val_100_AB.jpg')
G_model_file = osp.join(here, 'data/edges2shoes_net_G.pth')
E_model_file = osp.join(here, 'data/edges2shoes_net_E.pth')

gpu = args.gpu
nz = 8
output_nc = 3

print('GPU id: %d' % args.gpu)
print('G model: %s' % G_model_file)
print('E model: %s' % E_model_file)
print('Input file: %s' % img_file)

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

G = G_Unet_add_all(
    input_nc=1,
    output_nc=output_nc,
    nz=nz,
    num_downs=8,
    ngf=64,
    norm_layer=get_norm_layer('instance'),
    nl_layer=get_non_linearity('relu'),
    use_dropout=False,
    gpu_ids=[gpu],
    upsample='basic',
)
G.load_state_dict(torch.load(G_model_file))
G.eval()
if torch.cuda.is_available():
    G.cuda()

E = E_ResNet(
    input_nc=output_nc,
    output_nc=nz,
    ndf=64,
    n_blocks=5,
    norm_layer=get_norm_layer('instance'),
    nl_layer=get_non_linearity('lrelu'),
    gpu_ids=[gpu],
    vaeLike=True,
)
E.load_state_dict(torch.load(E_model_file))
E.eval()
if torch.cuda.is_available():
    E.cuda()

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

n_samples = 33
random_state = np.random.RandomState(0)
z_samples = random_state.normal(0, 1, (n_samples, nz)).astype(np.float32)

real_A = np.repeat(real_A, 3, axis=2)
viz = [real_A, real_B]
for i in range(1 + n_samples):
    if i == 0:
        def get_z(mu, logvar):
            std = logvar.mul(0.5).exp_()
            batchsize = std.size(0)
            nz = std.size(1)
            eps = torch.autograd.Variable(torch.randn(batchsize, nz)).cuda()
            return eps.mul(std).add_(mu)

        mu, logvar = E.forward(x_B)
        z = get_z(mu, logvar)
    else:
        z = torch.from_numpy(z_samples[i - 1][None]).cuda()
        z = torch.autograd.Variable(z, volatile=True)

    y = G(x_A, z)

    fake_B = y.data.cpu().numpy()[0].transpose(1, 2, 0)
    fake_B = ((fake_B + 1) / 2. * 255.).astype(np.uint8)

    viz.append(fake_B)
viz = fcn.utils.get_tile_image(viz)

out_file = osp.join(here, 'logs/infer_pytorch.png')
cv2.imwrite(out_file, viz[:, :, ::-1])
print('Saved file: %s' % out_file)
