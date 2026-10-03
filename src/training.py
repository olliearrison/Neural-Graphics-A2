import torch
import numpy as np
from utils import *
from rendering import * 
#* idk if this should be in helpers

#* arbitrary order fine for 2D?
depth_order = torch.arange(N, device=get_device())


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def base_config():
    return {
        'image': "images/test_HF_img.png", 
        'Gaussian counts': (256, 1024, 4096), #sizes of feature grids
        'feat_dim': 2, #number of features per grid
        'epochs': 2000, 
        'image_out': "images/Experiment_1_srgb_neural.png", #NOT YET IMPLEMENTED
    }



def train(N, H, W, target):
    # parameters (leaf tensors, requires_grad=True); a spread-out init, e.g.:
    mu     = torch.rand(N, 2) * torch.tensor([W, H])          # (N, 2)  spread across the image
    log_s  = torch.log(0.02 * max(H, W) * torch.ones(N, 2))   # (N, 2)  small blobs, log space
    theta  = torch.zeros(N)                                   # (N,)    rotation
    color  = torch.zeros(N, 3)                                # (N, 3)  sigmoid -> 0.5 gray
    op_raw = torch.full((N,), -2.0)                           # (N,)    sigmoid -> ~0.12 opacity

    opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)
    finalImg = None
    for step in range(100):
        Sigma = covariance_2d(log_s.exp(), theta)
        img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), torch.arange(N), H, W)
        finalImg = img
        loss  = ((img - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        # psnr = -10 * torch.log10(loss)
        if (step % 10 == 0): print("running epoch " + str(step))
        
    
    save_image(finalImg, "/Users/tunger/neural_graphics/Neural-Graphics-A2/results/final_img.png")

