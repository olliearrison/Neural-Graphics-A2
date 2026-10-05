
import torch
from utils import *

def render(mu, Sigma, color, opacity, order, H, W):
    xy = pixel_grid(H, W, device=mu.device)                     # (H*W, 2)
    w  = gaussian_weight(xy, mu, Sigma)       # (P, N)  from P1
    alpha = opacity[None, :] * w              # (P, N)
    C = torch.zeros(H * W, 3, device=mu.device)
    T = torch.ones(H * W, device=mu.device)
    # for i in order:                           # front to back
    #     a = alpha[:, i]
    #     C = C + (T * a)[:, None] * color[i]
    #     T = T * (1 - a)

    log_t = torch.log1p(-alpha.clamp(max=1 - 1e-6))
    T = torch.exp(torch.cumsum(log_t, dim = 1) - log_t)
    C = (T * alpha) @ color[order]
    # print(f"C_shape: {C.shape}")
    
    return C.reshape(H, W, 3)