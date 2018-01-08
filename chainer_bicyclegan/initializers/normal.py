from chainer import initializer
from chainer.initializers.normal import Normal
import numpy


class XavierNormal(initializer.Initializer):

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        super(XavierNormal, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(2. / (fan_in + fan_out))
        Normal(s)(array)
