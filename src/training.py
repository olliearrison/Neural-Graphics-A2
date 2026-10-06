import torch
import numpy as np
from utils import *
from rendering import * 
import math
import time


def base_config():
    return {
        'image': "images/test_HF_img.png", 
        'Gaussian counts': (256, 1024, 4096), #sizes of feature grids
        'feat_dim': 2, #number of features per grid
        'epochs': 2000, 
        'image_out': "images/Experiment_1_srgb_neural.png", #NOT YET IMPLEMENTED
    }

def train(N, H, W, target):
    dev = target.device

    # parameters (leaf tensors, requires_grad=True); a spread-out init, e.g.:
    mu     = (torch.rand(N, 2, device=dev) * torch.tensor([W, H], device=dev)).requires_grad_()         # (N, 2)  spread across the image
    log_s  = torch.full((N, 2), math.log(0.02 * max(H, W)), device=dev, requires_grad=True)   # (N, 2)  small blobs, log space
    theta  = torch.zeros(N, device=dev, requires_grad=True)                                   # (N,)    rotation
    color  = torch.zeros(N, 3, device=dev, requires_grad=True)                                # (N, 3)  sigmoid -> 0.5 gray
    op_raw = torch.full((N,), -2.0, device=dev, requires_grad=True)                           # (N,)    sigmoid -> ~0.12 opacity
    order = torch.arange(N, device=dev)

    opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)
    finalImg = None
    # currently taking sec per step with 1024 gaussians
    for step in range(2000):
        start_time = time.perf_counter()
        Sigma = covariance_2d(log_s.clamp(max=math.log(max(H, W) / 8)).exp(), theta)
        # print("got sigma")
        img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), order, H, W, tile=8, k_sigma=3)
        if (step % 100 == 0): print(step, "K", render.K, "MPS mem %.1f GB" % (torch.mps.driver_allocated_memory() / 1e9))
        # print("got image")
        finalImg = img
        loss  = ((img - target) ** 2).mean()
        # print("got loss")
        opt.zero_grad(); loss.backward(); opt.step()
        # psnr = -10 * torch.log10(loss)
        torch.mps.synchronize()
        end_time = time.perf_counter()

        #print(f"Step spent {end_time - start_time} seconds!")
        

    save_image(finalImg, f"results/final-coffee-{N}.png")

def train3d(N, train_cameras, iters, dev):
    # parameters (leaf tensors, requires_grad=True); example init for this scene:
    mu3    = ((torch.rand(N, 3, device=dev) * 2 - 1) * 1.5).requires_grad_()     # (N, 3)  cloud in ~[-1.5, 1.5]^3
    log_s  = torch.log(0.08 * torch.ones(N, 3, device=dev)).requires_grad_()   # (N, 3)  small 3D blobs
    quat   = torch.zeros(N, 4, device=dev); quat[:, 0] = 1.0; quat.requires_grad_()     # (N, 4)  identity rotation (w, x, y, z)
    color  = torch.zeros(N, 3, device=dev, requires_grad=True)                       # (N, 3)  sigmoid -> gray
    op_raw = torch.full((N,), -2.0, device=dev, requires_grad=True)                  # (N,)    sigmoid -> low opacity
    opt    = torch.optim.Adam([mu3, log_s, quat, color, op_raw], lr=1e-2)

    finalImg = None

    for step in range(iters):                        # e.g. N = 4000 Gaussians, iters = 1500
        cam   = random_choice(train_cameras, dev)
        Sig3  = covariance_3d(log_s.exp(), quat)            # 3D scale + rotation
        mu2, Sig2, depth = project_gaussian(mu3, Sig3, cam.R, cam.t, cam.K)
        order = torch.argsort(depth, descending=False)      # front-to-back: nearest (smallest z_c) first
        img   = render(mu2, Sig2, color.sigmoid(), op_raw.sigmoid(), order, cam.H, cam.W, tile=8, k_sigma=3)
        if (step % 100 == 0): print(step, "K", render.K, "MPS mem %.1f GB" % (torch.mps.driver_allocated_memory() / 1e9))
        finalImg = img
        loss  = ((img - cam.image) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    save_image(finalImg, f"results/final-spheres-{N}-{iters}.png")