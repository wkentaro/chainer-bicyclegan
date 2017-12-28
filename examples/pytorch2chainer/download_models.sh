#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FILE=edges2shoes

OUTPUT=$HERE/data/${FILE}_net_G.pth
URL=https://people.eecs.berkeley.edu/~junyanz/projects/BicycleGAN/models/${FILE}_net_G.pth
if [ ! -e $OUTPUT ]; then
  wget $URL -O $OUTPUT
fi

OUTPUT=$HERE/data/${FILE}_net_E.pth
URL=https://people.eecs.berkeley.edu/~junyanz/projects/BicycleGAN/models/${FILE}_net_E.pth
if [ ! -e $OUTPUT ]; then
  wget $URL -O $OUTPUT
fi
