import chainer
import chainer.functions as F
import chainer.links as L

from chainer_cyclegan.links import InstanceNormalization


class G_Unet_add_all(chainer.Chain):

    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, upsample='basic',
                 use_dropout=False):
        super(G_Unet_add_all, self).__init__()
        unet_block = UnetBlock_with_z(
            ngf * 8, ngf * 8, ngf * 8, nz, None,
            innermost=True, norm_layer=norm_layer, nl_layer=nl_layer,
            upsample=upsample)
        unet_block = UnetBlock_with_z(
            ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
            norm_layer=norm_layer, nl_layer=nl_layer,
            use_dropout=use_dropout, upsample=upsample)
        for i in range(num_downs - 6):
            unet_block = UnetBlock_with_z(
                ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                norm_layer=norm_layer, nl_layer=nl_layer,
                use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock_with_z(
            ngf * 4, ngf * 4, ngf * 8, nz, unet_block,
            norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(
            ngf * 2, ngf * 2, ngf * 4, nz, unet_block,
            norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(
            ngf, ngf, ngf * 2, nz, unet_block,
            norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(
            input_nc, output_nc, ngf, nz, unet_block,
            outermost=True,
            norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        with self.init_scope():
            self.model = unet_block

    def __call__(self, x, z):
        return self.model(x, z)


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


class UnetBlock_with_z(chainer.Chain):

    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 upsample='basic', padding_type='zero'):
        super(UnetBlock_with_z, self).__init__()
        assert padding_type == 'zero'
        p = 1

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv = [
            L.Convolution2D(input_nc, inner_nc, ksize=4, stride=2, pad=p),
        ]
        downrelu = lambda x: F.leaky_relu(x, slope=0.2)

        assert nl_layer == 'relu'
        uprelu = F.relu

        assert norm_layer in [None, 'instance']
        if norm_layer == 'instance':
            norm_layer = lambda size: InstanceNormalization(
                size, decay=0.9, eps=1e-05, use_beta=False, use_gamma=False)

        assert upsample == 'basic'
        if outermost:
            upconv = [
                L.Deconvolution2D(inner_nc * 2, outer_nc,
                                  ksize=4, stride=2, pad=1),
            ]
            down = downconv
            up = [uprelu] + upconv + [F.tanh]
        elif innermost:
            upconv = [
                L.Deconvolution2D(inner_nc, outer_nc,
                                  ksize=4, stride=2, pad=1),
            ]
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = [
                L.Deconvolution2D(inner_nc * 2, outer_nc,
                                  ksize=4, stride=2, pad=1),
            ]
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [F.dropout]

        with self.init_scope():
            self.down = Sequential(*down)
            self.submodule = submodule
            self.up = Sequential(*up)

    def __call__(self, x, z):
        if self.nz > 0:
            z_img = F.reshape(z, (z.shape[0], z.shape[1], 1, 1))
            z_img = F.tile(z_img, (1, 1, x.shape[2], x.shape[3]))
            x_and_z = F.concat([x, z_img], axis=1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return F.concat([x1, x], axis=1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return F.concat([self.up(x2), x], axis=1)


# -----------------------------------------------------------------------------


class BasicBlock(chainer.Chain):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        assert norm_layer == 'instance'
        norm_layer = lambda size: InstanceNormalization(
            size, decay=0.9, eps=1e-05, use_beta=False, use_gamma=False)
        assert nl_layer == 'lrelu'
        nl_layer_func = F.leaky_relu

        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer_func]
        layers += [
            L.Convolution2D(inplanes, inplanes, ksize=3, stride=1, pad=1)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer_func]
        layers += [Sequential(
            L.Convolution2D(inplanes, outplanes, ksize=3, stride=1, pad=1),
            lambda x: F.average_pooling_2d(x, ksize=2, stride=2),
        )]
        with self.init_scope():
            self.conv = Sequential(*layers)
            self.shortcut = Sequential(
                lambda x: F.average_pooling_2d(x, ksize=2, stride=2),
                L.Convolution2D(inplanes, outplanes, ksize=1, stride=1, pad=0),
            )

    def __call__(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class E_ResNet(chainer.Chain):

    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, gpu_ids=[], vaeLike=False):
        super(E_ResNet, self).__init__()

        assert nl_layer == 'lrelu'
        nl_layer_func = F.leaky_relu

        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            L.Convolution2D(input_nc, ndf, ksize=4, stride=2, pad=1)
        ]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n + 1)  # 2**n
            conv_layers += [
                BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer),
            ]
        conv_layers += [
            nl_layer_func,
            lambda x: F.average_pooling_2d(x, ksize=8),
        ]
        with self.init_scope():
            if vaeLike:
                self.fc = Sequential(*[L.Linear(output_ndf, output_nc)])
                self.fcVar = Sequential(*[L.Linear(output_ndf, output_nc)])
            else:
                self.fc = Sequential(*[L.Linear(output_ndf, output_nc)])
            self.conv = Sequential(*conv_layers)

    def __call__(self, x):
        x_conv = self.conv(x)
        conv_flat = F.reshape(x_conv, (x.shape[0], -1))
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output
