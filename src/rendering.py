
import torch
from utils import *

def render(mu, Sigma, color, opacity, order, H, W):
    # color: (N, 3),  opacity: (N,) in [0, 1],  order: indices sorted front -> back
    xy = pixel_grid(H, W)                     # (H*W, 2)
    w  = gaussian_weight(xy, mu, Sigma)       # (P, N)  from P1
    alpha = opacity[None, :] * w              # (P, N)
    C = torch.zeros(H * W, 3)
    T = torch.ones(H * W)
    for i in order:                           # front to back
        a = alpha[:, i]
        C = C + T * a * color[i]
        T = T * (1 - a)


    return C.reshape(H, W, 3)