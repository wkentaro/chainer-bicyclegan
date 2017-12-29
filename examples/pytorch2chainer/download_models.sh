#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT=$HERE/data/edges2shoes_net_E.pth
URL=https://people.eecs.berkeley.edu/~junyanz/projects/BicycleGAN/models/edges2shoes_net_E.pth
if [ ! -e $OUTPUT ]; then
  wget $URL -O $OUTPUT
fi

OUTPUT=$HERE/data/edges2shoes_net_G.pth
URL=https://people.eecs.berkeley.edu/~junyanz/projects/BicycleGAN/models/edges2shoes_net_G.pth
if [ ! -e $OUTPUT ]; then
  wget $URL -O $OUTPUT
fi

OUTPUT=$HERE/data/edges2shoes_net_E_from_chainer.npz
URL=https://drive.google.com/uc?id=1Qr_F3HkaTipqqpBBkuxJa3vLJea6bJ2C
if [ ! -e $OUTPUT ]; then
  pip install -q gdown
  gdown $URL -O $OUTPUT
fi

OUTPUT=$HERE/data/edges2shoes_net_G_from_chainer.npz
URL=https://drive.google.com/uc?id=1xAKLjtszp1d0AklVIPK_HqRjuYncDRuE
if [ ! -e $OUTPUT ]; then
  pip install -q gdown
  gdown $URL -O $OUTPUT
fi
