#!/usr/bin/env python

import argparse

import cv2
import numpy as np

from chainer_cyclegan.datasets import BerkeleyPix2PixDataset


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--split', default='train',
                        choices=['train', 'val'], help='dataset split')
    args = parser.parse_args()

    dataset = BerkeleyPix2PixDataset('edges2shoes', split=args.split)
    for i in range(len(dataset)):
        img_A, img_B = dataset[i]

        viz = np.hstack([img_A, img_B])

        cv2.imshow(__file__, viz[:, :, ::-1])
        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == '__main__':
    main()
