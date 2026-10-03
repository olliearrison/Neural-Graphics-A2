
import torch
from utils import *

def render(mu, Sigma, color, opacity, order, H, W):
    print(order.size(0))
    xy = pixel_grid(H, W)                     # (H*W, 2)
    w  = gaussian_weight(xy, mu, Sigma)       # (P, N)  from P1
    alpha = opacity[None, :] * w              # (P, N)
    C = torch.zeros(H * W, 3)
    T = torch.ones(H * W)
    for i in order:                           # front to back
        a = alpha[:, i]
        C = C + (T * a)[:, None] * color[i]
        T = T * (1 - a)

    print(f"C_shape: {C.shape}")
    return C.reshape(H, W, 3)