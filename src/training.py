import torch
from utils import *
from rendering import *

# parameters (leaf tensors, requires_grad=True); a spread-out init, e.g.:
mu     = torch.rand(N, 2) * torch.tensor([W, H])          # (N, 2)  spread across the image
log_s  = torch.log(0.02 * max(H, W) * torch.ones(N, 2))   # (N, 2)  small blobs, log space
theta  = torch.zeros(N)                                   # (N,)    rotation
color  = torch.zeros(N, 3)                                # (N, 3)  sigmoid -> 0.5 gray
op_raw = torch.full((N,), -2.0)                           # (N,)    sigmoid -> ~0.12 opacity

#* arbitrary order fine for 2D?
depth_order = torch.arange(N, device=get_device())

opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)
for step in range(2000):
    Sigma = covariance_2d(log_s.exp(), theta)
    img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), depth_order, H, W)
    loss  = ((img - target) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    # psnr = -10 * torch.log10(loss)