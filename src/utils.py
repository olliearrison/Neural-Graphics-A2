import numpy as np
import torch
from PIL import Image
import io
import random

# H = 100
# W = 100
# N = 50
# log_s  = torch.log(0.02 * max(H, W) * torch.ones(N, 2))   # (N, 2)  small blobs, log space
# theta  = torch.zeros(N)                                   # (N,)    rotation

class Camera:
    def __init__(self, H, W, image, R, t, K):
        self.H = H
        self.W = W
        self.image = image
        self.R = R
        self.t = t
        self.K = K

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

def quaternion_to_rotation(q):
    # q: (N, 4) as (w, x, y, z)
    # TODO: normalize q, then build R(q) above          -> (N, 3, 3)

    q = q / q.norm(dim = -1, keepdim = True)

    wVals = q[:, 0]
    xVals = q[:, 1]
    yVals = q[:, 2]
    zVals = q[:, 3]

    topRow = torch.stack([1 - 2 * (yVals * yVals + zVals * zVals),  
                          2 * (xVals*yVals - wVals*zVals),
                          2 * (xVals*zVals + wVals*yVals)], -1)
    
    midRow = torch.stack([2 * (xVals*yVals + wVals*zVals), 
                          1 - 2 * (xVals*xVals + zVals*zVals),
                          2 * (yVals*zVals - wVals*xVals)], -1)

    botRow = torch.stack([2 * (xVals*zVals - wVals*yVals),
                          2 * (yVals*zVals + wVals*xVals),
                          1 - 2 * (xVals*xVals + yVals*yVals)], -1)

    return torch.stack([topRow, midRow, botRow], -2)
    
def covariance_3d(scale, quat):
    # scale: (N, 3) positive,  quat: (N, 4)
    # TODO: R = quaternion_to_rotation(quat); return R S S^T R^T  -> (N, 3, 3)

    R = quaternion_to_rotation(quat)

    RS = R * scale[:, None, :]

    return RS @ RS.mT

def project_gaussian(mu3, Sigma3, R_wc, t, K):
    # mu3: (N, 3) world means,  Sigma3: (N, 3, 3) world covariances
    mu_cam = mu3 @ R_wc.T + t                  # world -> camera
    # TODO: mu2   = perspective-project mu_cam with K            (N, 2)
    # TODO: J     = Jacobian of the projection at mu_cam         (N, 2, 3)
    # TODO: Scam  = R_wc @ Sigma3 @ R_wc.T                       (N, 3, 3)
    #       Sig2 = J @ Scam @ J.transpose(-1, -2)                (N, 2, 2)

    temp = mu_cam @ K.T
    mu2 = temp[:, :2] / temp[:, 2:3]

    fx = K[0, 0]
    fy = K[1, 1]
    xc = mu_cam[:, 0]
    yc = mu_cam[:, 1]
    zc = mu_cam[:, 2]

    
    J = torch.stack([torch.stack([fx / zc, torch.zeros_like(zc), -fx*xc/torch.pow(zc, 2)], -1),
                     torch.stack([torch.zeros_like(zc), fy / zc, -fy*yc/torch.pow(zc, 2)], -1)], -2)

    Scam = R_wc @ Sigma3 @ R_wc.T
    Sig2 = J @ Scam @ J.transpose(-1, -2)

    depth = mu_cam[:, 2]
    return mu2, Sig2, depth

def random_choice(train_cameras, dev):
    Kval = torch.tensor(train_cameras['K'], device=dev)
    H = train_cameras['height']
    W = train_cameras['width']


    choices = train_cameras['frames']

    cam = random.choice(choices)

    image = image_to_tensor("data/spheres/" + cam['file'])
    Rval = torch.tensor(cam['R_wc'], device=dev)
    tval = torch.tensor(cam['t'], device=dev)

    return Camera(H=H, W=W, image=image, R=Rval, t=tval, K=Kval)


