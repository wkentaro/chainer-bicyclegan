#!/usr/bin/env python

import os.path as osp
import sys

import chainer
import chainer.links as L
from chainer_cyclegan.links import InstanceNormalization
import numpy as np
import torch

import cv2  # NOQA

here = osp.dirname(osp.abspath(__file__))
pytorch_dir = osp.realpath(osp.join(here, '../../src/pytorch-bicyclegan'))
sys.path.insert(0, pytorch_dir)

from models.networks import D_NLayersMulti
from models.networks import get_norm_layer

import chainer_bicyclegan


def convert_D(D_model_file, out_file):
    output_nc = 3

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

    D_chainer = chainer_bicyclegan.models.D_NLayersMulti(
        input_nc=output_nc,
        ndf=64,
        n_layers=3,
        norm_layer='instance',
        use_sigmoid=False,
        num_D=2,
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
            elif isinstance(l2, chainer_bicyclegan.models.BasicBlock):
                l2_list = l2.conv.functions
                l1_list = l1.conv
                copyto(l2_list, l1_list)
                l2_list = l2.shortcut.functions
                l1_list = l1.shortcut
                copyto(l2_list, l1_list)
            elif isinstance(l2, chainer_bicyclegan.models.Sequential):
                l2_list = l2.functions
                l1_list = l1
                copyto(l2_list, l1_list)
            else:
                print('Skip: {} -> {}'.format(type(l1), type(l2)))
                continue
            print('Copy: {} -> {}'.format(type(l1), type(l2)))

    copyto(D_chainer.model.functions, D.model)

    chainer.serializers.save_npz(out_file, D_chainer)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    params = []
    for param in D.parameters():
        params.append(param.data.numpy().flatten())
    params = np.hstack(params)
    print(params.min(), params.mean(), params.max())
    print('==========================================================')
    params = []
    for param in D_chainer.params():
        params.append(param.array.flatten())
    params = np.hstack(params)
    print(params.min(), params.mean(), params.max())
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


def main():
    D_model_file = osp.join(here, 'data/edges2shoes_net_D.pth')
    out_file = osp.join(here, 'data/edges2shoes_net_D_from_chainer.npz')
    convert_D(D_model_file, out_file)

    D_model_file = osp.join(here, 'data/edges2shoes_net_D2.pth')
    out_file = osp.join(here, 'data/edges2shoes_net_D2_from_chainer.npz')
    convert_D(D_model_file, out_file)


if __name__ == '__main__':
    main()
