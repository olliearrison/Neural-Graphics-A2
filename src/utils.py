import numpy as np
import torch
from PIL import Image
import io

# H = 100
# W = 100
# N = 50
# log_s  = torch.log(0.02 * max(H, W) * torch.ones(N, 2))   # (N, 2)  small blobs, log space
# theta  = torch.zeros(N)                                   # (N,)    rotation

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def covariance_2d(scale, theta):
    # scale: (N, 2) positive,  theta: (N,) radians
    # TODO: build R(theta) and S = diag(scale), return Sigma = R S S^T R^T  -> (N, 2, 2)
    cosVals = torch.cos(theta)
    sinVals = torch.sin(theta)

    R = torch.stack([torch.stack([cosVals, -sinVals], -1),
                     torch.stack([sinVals, cosVals], -1)], -2)

    RS = R * scale[:, None, :]

    return RS @ RS.mT


def gaussian_weight(xy, mu, Sigma):
    # xy: (P, 2) pixel coords,  mu: (N, 2),  Sigma: (N, 2, 2)
    # TODO: w[p, n] = exp(-0.5 (xy_p - mu_n)^T Sigma_n^-1 (xy_p - mu_n))
    #                          # (P, N)

    # for n in range(N):
    #     for p in range(P):
    #         d = xy[p] - mu[n]
    #         #* Documentation suggests using solve for numerical stability 
    #         #* should be equiv to A.inv() @ B
    #         #? https://docs.pytorch.org/docs/2.14/generated/torch.linalg.inv.html
    #         solved = torch.linalg.solve(Sigma[n], d)
    #         dist_sq = d @ solved
    #         result[p, n] = torch.exp(-0.5 * dist_sq)

    d = xy[:, None, :] - mu[None, :, :]
    invSig = torch.linalg.inv(Sigma)
    dist_sq = torch.einsum('pni, nij, pnj->pn', d, invSig, d)
    result = torch.exp(-0.5 * dist_sq)

    return result


def pixel_grid(H, W, device = None):
    ys, xs = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device),
                            indexing='ij')
    return torch.stack([xs, ys], -1).reshape(-1, 2).float() + 0.5

#* (H, W, 3), float32, with values in [0, 1]
def image_to_tensor(path):
    device = get_device()

    with Image.open(path) as im:
        pixels = np.array(im.convert("RGB"), dtype=np.float32) / 255.0

    return torch.from_numpy(pixels).to(device)

#* (H, W, 3) to [0, 1]
def save_image(image, path):
    pixels = (
        image.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    Image.fromarray(pixels).save(path)

