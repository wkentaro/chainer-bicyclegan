#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

download()
{
  if [ ! $# -eq 2 ]; then
    return 1
  fi
  url=$1
  output=$2
  if [ ! -e $output ]; then
    pip install -q gdown
    gdown $url -O $output
  fi
}

# ---------------------------------------------------------------------

OUTPUT=$HERE/data/edges2shoes_net_E.pth
URL=https://people.eecs.berkeley.edu/~junyanz/projects/BicycleGAN/models/edges2shoes_net_E.pth
download $URL $OUTPUT

OUTPUT=$HERE/data/edges2shoes_net_G.pth
URL=https://people.eecs.berkeley.edu/~junyanz/projects/BicycleGAN/models/edges2shoes_net_G.pth
download $URL $OUTPUT

OUTPUT=$HERE/data/edges2shoes_net_D.pth
URL=https://drive.google.com/uc?id=17GXh5bk3604dTTeAIZ66lBfPSXMCnqe9
download $URL $OUTPUT

OUTPUT=$HERE/data/edges2shoes_net_D2.pth
URL=https://drive.google.com/uc?id=1X3Ipu_y14ZNdErAKc0rRpETrqSqUrSiV
download $URL $OUTPUT

# ---------------------------------------------------------------------

OUTPUT=$HERE/data/edges2shoes_net_E_from_pytorch.npz
URL=https://drive.google.com/uc?id=1Qr_F3HkaTipqqpBBkuxJa3vLJea6bJ2C
download $URL $OUTPUT

OUTPUT=$HERE/data/edges2shoes_net_G_from_pytorch.npz
URL=https://drive.google.com/uc?id=1xAKLjtszp1d0AklVIPK_HqRjuYncDRuE
download $URL $OUTPUT

OUTPUT=$HERE/data/edges2shoes_net_D_from_pytorch.npz
URL=https://drive.google.com/uc?id=13EgBs6quBybwzKJaGMz6iDWAg1hU0KMZ
download $URL $OUTPUT

OUTPUT=$HERE/data/edges2shoes_net_D2_from_pytorch.npz
URL=https://drive.google.com/uc?id=1rZFEi8ibwIhPcNs9F95EVoxD11IZFAw3
download $URL $OUTPUT

# ---------------------------------------------------------------------

OUTPUT=$HERE/data/edges2shoes_net_E_20180102_024055.npz
URL=https://drive.google.com/uc?id=1uyAkI0M_UJLluFBiOVoPglKmiHnl5prf
download $URL $OUTPUT

OUTPUT=$HERE/data/edges2shoes_net_G_20180102_024055.npz
URL=https://drive.google.com/uc?id=1wYYtkWUhcW_oVw5WTzojVs2_HsVRggUH
download $URL $OUTPUT

OUTPUT=$HERE/data/edges2shoes_net_D_20180102_024055.npz
URL=https://drive.google.com/uc?id=16fkqeGzpNc-MRhZw1kiMe59E4bfmhNR2
download $URL $OUTPUT

OUTPUT=$HERE/data/edges2shoes_net_D2_20180102_024055.npz
URL=https://drive.google.com/uc?id=1lr6xKtM8CM-rSpuXlyiWVMWJQ_7wTErO
download $URL $OUTPUT
