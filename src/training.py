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
    # currently taking 1.4sec per step with 1024 gaussians
    for step in range(2000):
        start_time = time.perf_counter()
        if (step % 100 == 0): print("running epoch " + str(step))
        Sigma = covariance_2d(log_s.exp(), theta)
        # print("got sigma")
        img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), order, H, W)
        # print("got image")
        finalImg = img
        loss  = ((img - target) ** 2).mean()
        # print("got loss")
        opt.zero_grad(); loss.backward(); opt.step()
        # psnr = -10 * torch.log10(loss)
        end_time = time.perf_counter()

        print(f"Step spent {end_time - start_time} seconds!")
        

    save_image(finalImg, "/results/final_img.png")

