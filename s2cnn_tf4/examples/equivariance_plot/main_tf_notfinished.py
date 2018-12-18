# pylint: disable=C,R,E1101,E1102
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

import torch
from s2cnn import so3_rotation


def s2_rotation(x, a, b, c):
    x = so3_rotation(x.view(*x.size(), 1).expand(*x.size(), x.size(-1)), a, b, c)
    return x[..., 0]


def plot(x, text, normalize=False):
    assert x.size(0) == 1
    assert x.size(1) in [1, 3]
    x = x[0]
    if x.dim() == 4:
        x = x[..., 0]

    nch = x.size(0)
    is_rgb = (nch == 3)

    if normalize:
        x = x - x.view(nch, -1).mean(-1).view(nch, 1, 1)
        x = 0.4 * x / x.view(nch, -1).std(-1).view(nch, 1, 1)

    x = x.detach().cpu().numpy()
    x = x.transpose((1, 2, 0)).clip(0, 1)

    print(x.shape)
    if is_rgb:
        plt.imshow(x)
    else:
        plt.imshow(x[:, :, 0], cmap='gray')
    plt.axis("off")

    plt.text(0.5, 0.5, text,
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             color='white', fontsize=20)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load image
    x = imread("earth128.jpg").astype(np.float32).transpose((2, 0, 1)) / 255
    b = 64
    x = torch.tensor(x, dtype=torch.float, device=device)
    x = x.view(1, 3, 2 * b, 2 * b)


    # test equivariance
    abc = (0.5, 1, 0)  # rotation angles

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plot(x, "x : signal on the sphere")

    plt.subplot(1, 2, 2)
    plot(s2_rotation(x, *abc), "R(x) : rotation using fft")

    plt.tight_layout()
    plt.savefig("figma.jpeg")


if __name__ == "__main__":
    main()
