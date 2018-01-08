import chainercv
import numpy as np
import PIL.Image


class BicycleGANTransform(object):

    def __init__(self, train=True, load_size=(286, 286), fine_size=(256, 256)):
        self._train = train
        self._load_size = load_size
        self._fine_size = fine_size

    def __call__(self, in_data):
        img_A, img_B = in_data

        img_A = img_A.transpose(2, 0, 1)
        img_A = chainercv.transforms.resize(
            img_A, size=self._load_size,
            interpolation=PIL.Image.BICUBIC)
        img_A, param_crop = chainercv.transforms.random_crop(
            img_A, size=self._fine_size, return_param=True)
        if self._train:
            # img_A, param_flip = chainercv.transforms.random_flip(
            #     img_A, x_random=True, return_param=True)
            pass
        img_A = img_A.astype(np.float32) / 255  # ToTensor
        img_A = (img_A - 0.5) / 0.5  # Normalize

        img_B = img_B.transpose(2, 0, 1)
        img_B = chainercv.transforms.resize(
            img_B, size=self._load_size,
            interpolation=PIL.Image.BICUBIC)
        img_B = img_B[:, param_crop['y_slice'], param_crop['x_slice']]
        if self._train:
            # img_B = chainercv.transforms.flip(
            #     img_B, x_flip=param_flip['x_flip'])
            pass
        img_B = img_B.astype(np.float32) / 255  # ToTensor
        img_B = (img_B - 0.5) / 0.5  # Normalize

        return img_A, img_B
