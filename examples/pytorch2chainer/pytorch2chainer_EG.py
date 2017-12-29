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

from models.networks import E_ResNet
from models.networks import G_Unet_add_all
from models.networks import get_non_linearity
from models.networks import get_norm_layer

import lib


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

    G_chainer = lib.models.G_Unet_add_all(
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


def convert_E(nz, output_nc):
    E_model_file = osp.join(here, 'data/edges2shoes_net_E.pth')
    E = E_ResNet(
        input_nc=output_nc,
        output_nc=nz,
        ndf=64,
        n_blocks=5,
        norm_layer=get_norm_layer('instance'),
        nl_layer=get_non_linearity('lrelu'),
        gpu_ids=[],
        vaeLike=True,
    )
    E.load_state_dict(torch.load(E_model_file))

    E_chainer = lib.models.E_ResNet(
        input_nc=output_nc,
        output_nc=nz,
        ndf=64,
        n_blocks=5,
        norm_layer='instance',
        nl_layer='lrelu',
        vaeLike=True,
    )

    def copyto(l2_list, l1_list):
        assert len(l2_list) == len(l1_list)
        for l1, l2 in zip(l1_list, l2_list):
            if isinstance(l2, (L.Convolution2D, L.Deconvolution2D, L.Linear)):
                np.copyto(l2.W.array, l1.weight.data.numpy())
                np.copyto(l2.b.array, l1.bias.data.numpy())
            elif isinstance(l2, InstanceNormalization):
                np.copyto(l2.avg_mean, l1.running_mean.numpy())
                np.copyto(l2.avg_var, l1.running_var.numpy())
            elif isinstance(l2, lib.models.BasicBlock):
                l2_list = l2.conv.functions
                l1_list = l1.conv
                copyto(l2_list, l1_list)
            elif isinstance(l2, lib.models.Sequential):
                l2_list = l2.functions
                l1_list = l1
                copyto(l2_list, l1_list)
            else:
                print('Skip: {} -> {}'.format(type(l1), type(l2)))
                continue
            print('Copy: {} -> {}'.format(type(l1), type(l2)))

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(E.fc)
    l1_list = E.fc
    l2_list = E_chainer.fc.functions
    copyto(l2_list, l1_list)
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(E.fcVar)
    l1_list = E.fcVar
    l2_list = E_chainer.fcVar.functions
    copyto(l2_list, l1_list)
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(E.conv)
    l1_list = E.conv
    l2_list = E_chainer.conv.functions
    copyto(l2_list, l1_list)
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    out_file = osp.join(here, 'data/edges2shoes_net_E_from_chainer.npz')
    chainer.serializers.save_npz(out_file, E_chainer)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    params = []
    for param in E.parameters():
        params.append(param.data.numpy().flatten())
    params = np.hstack(params)
    print(params.min(), params.mean(), params.max())
    print('==========================================================')
    params = []
    for param in E_chainer.params():
        params.append(param.array.flatten())
    params = np.hstack(params)
    print(params.min(), params.mean(), params.max())
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


def main():
    nz = 8
    output_nc = 3
    convert_E(nz, output_nc)
    convert_G(nz, output_nc)


if __name__ == '__main__':
    main()
