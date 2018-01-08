import chainer

from chainer_cyclegan.links import InstanceNormalization


class Sequential(chainer.ChainList):

    def __init__(self, *functions):
        super(Sequential, self).__init__()
        self.functions = functions
        for func in functions:
            if isinstance(func, chainer.Link):
                self.add_link(func)

    def __call__(self, x):
        h = x
        for func in self.functions:
            if isinstance(func, InstanceNormalization):
                with chainer.using_config('train', True):
                    h = func(h)
            else:
                h = func(h)
        return h
