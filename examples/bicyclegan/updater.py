import chainer
from chainer import cuda
import chainer.functions as F
from chainer import training
from chainer import Variable
import numpy as np


def get_z_random(size0, size1):
    z_random = np.random.normal(0, 1, (size0, size1))
    z_random = z_random.astype(np.float32)
    z_random = cuda.to_gpu(z_random)
    return Variable(z_random)


class BicycleGANUpdater(training.StandardUpdater):

    def backward_D(self, D, real, fake):
        xp = cuda.get_array_module(real.array)

        # Real
        pred_real = D(real)
        loss_D_real = 0
        for pr in pred_real:
            loss_D_real += F.mean_squared_error(pr, xp.ones_like(pr.array))

        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = D(fake.array)
        loss_D_fake = 0
        for pf in pred_fake:
            loss_D_fake += F.mean_squared_error(pf, xp.zeros_like(pf.array))

        # Combined loss
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        loss_D.unchain_backward()

        chainer.report({
            'loss_D': loss_D,
        })

    def loss_G_GAN(self, fake, D, lambda_GAN):
        xp = cuda.get_array_module(fake.array)

        if lambda_GAN > 0.0:
            pred_fake = D(fake)
            loss_G_GAN = 0
            for pf in pred_fake:
                loss_G_GAN += F.mean_squared_error(pf, xp.ones_like(pf.array))
        else:
            loss_G_GAN = 0
        return loss_G_GAN * lambda_GAN

    def backward_EG(self, fake_data_encoded, fake_data_random,
                    fake_B_encoded, real_B_encoded,
                    D, D2, lambda_GAN, lambda_GAN2, mu, logvar):
        lambda_kl = 0.01
        lambda_L1 = 10.0

        # 1, G(A) should fool D
        loss_G_GAN = self.loss_G_GAN(fake_data_encoded, D, lambda_GAN)
        loss_G_GAN2 = self.loss_G_GAN(fake_data_random, D2, lambda_GAN2)
        # 2. KL loss
        if lambda_kl > 0:
            kl_element = (((mu ** 2) + F.exp(logvar)) * -1) + 1 + logvar
            loss_kl = F.sum(kl_element) * -0.5 * lambda_kl
        else:
            loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if lambda_L1 > 0:
            loss_G_L1 = F.mean_absolute_error(fake_B_encoded, real_B_encoded)
            loss_G_L1 = lambda_L1 * loss_G_L1
        else:
            loss_G_L1 = 0

        loss_G = loss_G_GAN + loss_G_GAN2 + loss_G_L1 + loss_kl
        loss_G.backward()  # Not unchain_backward for backward_G_alone

        chainer.report({
            'loss_G_GAN': loss_G_GAN,
            'loss_G_GAN2': loss_G_GAN2,
            'loss_G_L1': loss_G_L1,
            'loss_kl': loss_kl,
            'loss_G': loss_G,
        })

    def backward_G_alone(self, lambda_z, mu2, z_random):
        # 3, reconstruction |z_predit-z_random|
        if lambda_z > 0.0:
            loss_z_L1 = F.mean(F.absolute(mu2 - z_random)) * lambda_z
            loss_z_L1.backward()
            loss_z_L1.unchain_backward()
        else:
            loss_z_L1 = 0

        chainer.report({
            'loss_z_L1': loss_z_L1,
        })

    def update_core(self):
        optimizer_E = self.get_optimizer('E')
        optimizer_G = self.get_optimizer('G')
        optimizer_D = self.get_optimizer('D')
        optimizer_D2 = self.get_optimizer('D2')

        E = optimizer_E.target
        G = optimizer_G.target
        D = optimizer_D.target
        D2 = optimizer_D2.target

        batch = next(self.get_iterator('main'))

        if len(batch) != 2:
            return

        img_A, img_B = zip(*batch)
        img_A = np.asarray(img_A)[:, 0:1, :, :]
        img_B = np.asarray(img_B)

        assert len(img_A) == len(img_B) == 2
        real_A_encoded = Variable(self.converter(img_A[0:1], self.device))
        real_A_random = Variable(self.converter(img_A[1:2], self.device))
        real_B_encoded = Variable(self.converter(img_B[0:1], self.device))
        real_B_random = Variable(self.converter(img_B[1:2], self.device))

        # update D
        # -----------------------------------------------------------------
        mu, logvar = E(real_B_encoded)
        std = F.exp(logvar * 0.5)
        eps = get_z_random(std.shape[0], std.shape[1])
        z_encoded = (eps * std) + mu

        z_random = get_z_random(real_A_random.shape[0], std.shape[1])

        fake_B_encoded = G(real_A_encoded, z_encoded)

        # generate fake_B_random
        fake_B_random = G(real_A_encoded, z_random)

        mu2, logvar2 = E(fake_B_random)
        # std2 = F.exp(logvar2 * 0.5)
        # eps2 = get_z_random(std2.shape[0], std2.shape[1])
        # z_predict = (eps2 * std2) + mu2

        fake_data_encoded = fake_B_encoded
        fake_data_random = fake_B_random
        real_data_encoded = real_B_encoded
        real_data_random = real_B_random

        # update D1
        lambda_GAN = 1.0
        lambda_GAN2 = 1.0
        if lambda_GAN > 0:
            D.cleargrads()
            self.backward_D(D, real_data_encoded, fake_data_encoded)
            optimizer_D.update()

        # update D2
        if lambda_GAN2 > 0:
            D2.cleargrads()
            self.backward_D(D2, real_data_random, fake_data_random)
            optimizer_D2.update()

        # update G
        # -----------------------------------------------------------------
        E.cleargrads()
        G.cleargrads()
        self.backward_EG(
            fake_data_encoded, fake_data_random,
            fake_B_encoded, real_B_encoded,
            D, D2, lambda_GAN, lambda_GAN2,
            mu, logvar)
        optimizer_G.update()
        optimizer_E.update()

        # update G only
        lambda_z = 0.5
        if lambda_z > 0.0:
            G.cleargrads()
            E.cleargrads()
            self.backward_G_alone(lambda_z, mu2, z_random)
            optimizer_G.update()
