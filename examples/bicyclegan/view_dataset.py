#!/usr/bin/env python

import cv2
import numpy as np

from chainer_cyclegan.datasets import BerkeleyPix2PixDataset


dataset = BerkeleyPix2PixDataset('edges2shoes', split='train')
for i in range(len(dataset)):
    img_A, img_B = dataset[i]

    viz = np.hstack([img_A, img_B])

    cv2.imshow(__file__, viz[:, :, ::-1])
    if cv2.waitKey(0) == ord('q'):
        break
