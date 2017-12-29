# pytorch2chainer


## Usage

```bash
./download_models.sh

# to convert .pth -> .npz
# ./pytorch2chainer_E.py
# ./pytorch2chainer_G.py

./infer_pytorch.py --gpu 0
./infer_chainer.py --gpu 0
```

<table>
  <tr>
    <th>PyTorch</th><th>Chainer</th>
  </tr>
  <tr>
    <td><img src=".readme/infer_pytorch.jpg"></td>
    <td><img src=".readme/infer_chainer.jpg"></td>
  </tr>
</table>
