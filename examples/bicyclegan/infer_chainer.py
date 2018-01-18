#!/usr/bin/env python

import glob
import os
import os.path as osp
import subprocess


def main():
    here = osp.dirname(osp.abspath(__file__))
    exe = osp.join(here, '../pytorch2chainer/infer_chainer.py')

    logs_dir = osp.join(here, 'logs')
    for log_dir in os.listdir(logs_dir):
        if 'edges2shoes' in log_dir:
            img_file = None
        elif 'edges2handbags' in log_dir:
            img_file = osp.expanduser('~/data/datasets/wkentaro/chainer-cyclegan/edges2handbags/val/100_AB.jpg')  # NOQA

        log_dir = osp.join(logs_dir, log_dir)
        E_files = glob.glob(osp.join(log_dir, 'E_*.npz'))
        E_files = sorted(E_files)
        G_files = glob.glob(osp.join(log_dir, 'G_*.npz'))
        G_files = sorted(G_files)
        for E_file, G_file in zip(E_files, G_files):
            assert osp.basename(E_file)[2:] == osp.basename(G_file)[2:]

            # 00000XXX.jpg
            out_file = osp.splitext(osp.basename(G_file)[2:])[0] + '.jpg'
            out_file = osp.join(log_dir, 'infer_chainer', out_file)

            if osp.exists(out_file):
                continue

            # print('E: %s' % E_file)
            # print('G: %s' % G_file)
            # print('out: %s' % out_file)

            cmd = '%s -E %s -G %s -o %s' % (exe, E_file, G_file, out_file)
            if img_file:
                cmd += ' -i %s' % img_file
            subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
