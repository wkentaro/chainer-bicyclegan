#!/usr/bin/env python

import os.path as osp
import sys

import chainer
import chainer.links as L
from chainer_cyclegan.links import InstanceNormalization
import numpy as np
import torch

torch.backends.cudnn.benchmark = True

import cv2  # NOQA

here = osp.dirname(osp.abspath(__file__))
pytorch_dir = osp.realpath(osp.join(here, '../../src/pytorch-bicyclegan'))
sys.path.insert(0, pytorch_dir)

from models.networks import G_Unet_add_all
from models.networks import get_non_linearity
from models.networks import get_norm_layer

import chainer_bicyclegan


def convert_G(nz, output_nc):
    G_model_file = osp.join(here, 'data/edges2shoes_net_G.pth')
    G = G_Unet_add_all(
        input_nc=1,
        output_nc=output_nc,
        nz=nz,
        num_downs=8,
        ngf=64,
        norm_layer=get_norm_layer('instance'),
        nl_layer=get_non_linearity('relu'),
        use_dropout=False,
        gpu_ids=[],
        upsample='basic',
    )
    G.load_state_dict(torch.load(G_model_file))

    G_chainer = chainer_bicyclegan.models.G_Unet_add_all(
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

    def copyto(l2_list, l1_list):
        assert len(l1_list) == len(l2_list)
        for l1, l2 in zip(l1_list, l2_list):
            if isinstance(l2, (L.Convolution2D, L.Deconvolution2D)):
                np.copyto(l2.W.array, l1.weight.data.numpy())
                np.copyto(l2.b.array, l1.bias.data.numpy())
            elif isinstance(l2, InstanceNormalization):
                np.copyto(l2.avg_mean, l1.running_mean.numpy())
                np.copyto(l2.avg_var, l1.running_var.numpy())
            else:
                print('Skip: {} -> {}'.format(type(l1), type(l2)))
                continue
            print('Copy: {} -> {}'.format(type(l1), type(l2)))

    unet = G.model
    unet_chainer = G_chainer.model
    while True:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(unet)
        l1_list = unet.down
        l2_list = unet_chainer.down.functions
        copyto(l2_list, l1_list)

        l1_list = unet.up
        l2_list = unet_chainer.up.functions
        copyto(l2_list, l1_list)
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

        if unet.submodule is None:
            assert unet_chainer.submodule is None
            break

        unet = unet.submodule
        unet_chainer = unet_chainer.submodule

    out_file = osp.join(here, 'data/edges2shoes_net_G_from_chainer.npz')
    chainer.serializers.save_npz(out_file, G_chainer)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    params = []
    for param in G.parameters():
        params.append(param.data.numpy().flatten())
    params = np.hstack(params)
    print(params.min(), params.mean(), params.max())
    print('==========================================================')
    params = []
    for param in G_chainer.params():
        params.append(param.array.flatten())
    params = np.hstack(params)
    print(params.min(), params.mean(), params.max())
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


def main():
    nz = 8
    output_nc = 3
    convert_G(nz, output_nc)


if __name__ == '__main__':
    main()
