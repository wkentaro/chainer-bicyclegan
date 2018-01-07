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

data_dir = osp.join(here, 'data')
default_G_model_file = osp.join(data_dir, 'edges2shoes_net_G.pth')
default_E_model_file = osp.join(data_dir, 'edges2shoes_net_E.pth')
default_img_file = osp.join(data_dir, 'edges2shoes_val_100_AB.jpg')
default_out_file = osp.join(here, 'logs/infer_pytorch.png')

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id')
parser.add_argument('-i', '--img-file', default=default_img_file,
                    help='image file')
parser.add_argument('-E', '--E-model-file', default=default_E_model_file,
                    help='E model file')
parser.add_argument('-G', '--G-model-file', default=default_G_model_file,
                    help='G model file')
parser.add_argument('-o', '--out-file', default=default_out_file,
                    help='Output file')
args = parser.parse_args()

gpu = args.gpu
img_file = args.img_file
G_model_file = args.G_model_file
E_model_file = args.E_model_file

# -----------------------------------------------------------------------------

print('GPU id: %d' % args.gpu)
print('G model: %s' % G_model_file)
print('E model: %s' % E_model_file)
print('Input file: %s' % img_file)

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

nz = 8
output_nc = 3

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
np.random.seed(0)
z_samples = np.random.normal(0, 1, (n_samples, nz)).astype(np.float32)

real_A = np.repeat(real_A, 3, axis=2)
viz = [real_A, real_B]
for i in range(1 + n_samples):
    if i == 0:
        def get_z(mu, logvar):
            std = logvar.mul(0.5).exp_()
            batchsize = std.size(0)
            nz = std.size(1)
            eps = np.random.normal(0, 1, (batchsize, nz)).astype(np.float32)
            eps = torch.autograd.Variable(
                torch.from_numpy(eps).cuda(), volatile=True)
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

cv2.imwrite(args.out_file, viz[:, :, ::-1])
print('Saved file: %s' % args.out_file)
