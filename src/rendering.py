
import torch
import math
from utils import *

def composite(alpha, col):
    log_t = torch.log1p(-alpha.clamp(max=1 - 1e-6))
    T = torch.exp(torch.cumsum(log_t, dim = -1) - log_t)
    C = (T * alpha) @ col

    return C

def render(mu, Sigma, color, opacity, order, H, W, tile, k_sigma):
    xy = pixel_grid(H, W, device=mu.device)                     # (H*W, 2)
    # w  = gaussian_weight(xy, mu, Sigma)       # (P, N)  from P1
    # alpha = opacity[None, :] * w              # (P, N)
    # C = torch.zeros(H * W, 3, device=mu.device)
    # T = torch.ones(H * W, device=mu.device)
    # # for i in order:                           # front to back
    # #     a = alpha[:, i]
    # #     C = C + (T * a)[:, None] * color[i]
    # #     T = T * (1 - a)

    # log_t = torch.log1p(-alpha.clamp(max=1 - 1e-6))
    # T = torch.exp(torch.cumsum(log_t, dim = 1) - log_t)
    # C = (T * alpha) @ color[order]
    # # print(f"C_shape: {C.shape}")


    dev = mu.device
    mu, Sigma, color, opacity = mu[order], Sigma[order], color[order], opacity[order]
    N = mu.shape[0]
    ty, tx = math.ceil(H / tile), math.ceil(W / tile)

    with torch.no_grad():
        a, b, c = Sigma[:, 0, 0], Sigma[:, 0, 1], Sigma[:, 1, 1]
        lam_max = 0.5 * (a + c) + torch.sqrt((0.5 * (a - c)) ** 2 + b * b)
        r = k_sigma * lam_max.sqrt()

        gy, gx = torch.meshgrid(torch.arange(ty, device=dev),
                                torch.arange(tx, device=dev), 
                                 indexing='ij')
        # top left corners
        t0 = torch.stack([gx, gy], -1).reshape(-1, 1, 2).float() * tile

        lo, hi = mu - r[:, None], mu + r[:, None]
        hit = ((hi[None] > t0) & (lo[None] < t0 + tile)).all(-1)

        K = max(int(hit.sum(1).max()), 1)
        K = min(N, ((K + 63) // 64) * 64)
        render.K = K
        key = torch.where(hit, torch.arange(N, device=dev), N)
        idx = key.sort(dim=1).values[:, :K]

    mu_p = torch.cat([mu, torch.zeros(1, 2, device=dev)])
    Sig_p = torch.cat([Sigma, torch.eye(2, device=dev)[None]])
    col_p = torch.cat([color, torch.zeros(1, 3, device=dev)])
    op_p = torch.cat([opacity, torch.zeros(1, device=dev)])
    m, S, col, op = mu_p[idx], Sig_p[idx], col_p[idx], op_p[idx]

    a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    det = a * c - b * b
    A, B, C = (c / det)[:, None], (-b / det)[:, None], (a / det)[:, None]

    py, px = torch.meshgrid(torch.arange(tile, device=dev), 
                            torch.arange(tile, device=dev),
                            indexing='ij')

    local = torch.stack([px, py], -1).reshape(1, -1, 2).float() + 0.5
    pix = t0 + local

    dx = pix[..., 0:1] - m[:, None, :, 0]
    dy = pix[..., 1:2] - m[:, None, :, 1]
    alpha = op[:, None, :] * torch.exp(-0.5 * (A*dx*dx + 2*B*dx*dy + C*dy*dy))

    outCol = composite(alpha, col)

    img = (outCol.reshape(ty, tx, tile, tile, 3)
                 .permute(0, 2, 1, 3, 4)
                 .reshape(ty * tile, tx * tile, 3))

    return img[:H, :W]