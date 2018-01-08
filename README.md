# chainer-bicyclegan

Chainer implementation of ["Toward Multimodal Image-to-Image Translation"](https://arxiv.org/abs/1711.11586).  
This is a faithful re-implementation of [the official PyTorch implementation](https://github.com/junyanz/BicycleGAN).


## Installation

```bash
git clone --recursive https://github.com/wkentaro/chainer-bicyclegan.git
cd chainer-bicyclegan

conda install -c menpo -y opencv
pip install .
```


## Training

```bash
cd examples/bicyclegan

# ./train.py --gpu <gpu_id>
./train.py --gpu 0
```
