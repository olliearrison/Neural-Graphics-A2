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

@torch.no_grad()
def densify(params, opt, grad_mag, budget, *, size_threshold,
            grad_threshold=2e-4, split_scale=1.6, prune_opacity=0.005,
            max_log_scale=None):
    """
    params: (mu, log_s, theta_or_quat, color, op_raw), in Adam order.
    grad_mag: MEAN position-gradient magnitude since the last pass.
    size_threshold: pixels in 2D, world units in 3D.

    Call after opt.step(). Returns replacement leaf tensors and updates opt.
    Assumes the single-group Adam used in your current training functions.
    """
    mu, log_s, rotation, color, op_raw = params

    if budget < 1 or split_scale <= 1:
        print("budget must be positive and split_scale must exceed 1")

    #* Match the scale clamp used by 2D renderer, if supplied.
    effective_log_s = (
        log_s if max_log_scale is None
        else log_s.clamp(max=max_log_scale)
    )
    scale = effective_log_s.exp()

    #* Prune low-opacity Gaussians first.
    keep = op_raw.sigmoid() >= prune_opacity
    if not keep.any():
        keep[op_raw.argmax()] = True  # Keep the model trainable.

    #* Handle an input that already exceeds the budget.
    kept = keep.nonzero(as_tuple=True)[0]
    if kept.numel() > budget:
        ranked = torch.argsort(grad_mag[kept], descending=True)
        keep.zero_()
        keep[kept[ranked[:budget]]] = True

    #* Both cloning and splitting increase the total count by exactly one.
    room = max(0, budget - int(keep.sum().item()))
    candidates = (
        keep & (grad_mag > grad_threshold)
    ).nonzero(as_tuple=True)[0]

    ranked = torch.argsort(grad_mag[candidates], descending=True)
    selected = candidates[ranked[:room]]

    small = scale[selected].amax(dim=-1) <= size_threshold
    clones = selected[small]
    splits = selected[~small]

    if keep.all() and selected.numel() == 0:
        return params

    # Keep clone parents; remove split parents.
    keep[splits] = False
    kept = keep.nonzero(as_tuple=True)[0]
    split_parents = splits.repeat_interleave(2)

    #* Output layout: survivors, new clones, new split children.
    source = torch.cat([kept, clones, split_parents])
    values = [p[source].clone() for p in params]

    if split_parents.numel():
        # Sample offsets in the parent's local coordinate system.
        local = (
            torch.randn_like(scale[split_parents])
            * scale[split_parents]
        )

        if mu.shape[1] == 2:
            angle = rotation[split_parents]
            c, s = angle.cos(), angle.sin()
            offset = torch.stack([
                c * local[:, 0] - s * local[:, 1],
                s * local[:, 0] + c * local[:, 1],
            ], dim=-1)
        else:
            R = quaternion_to_rotation(rotation[split_parents])
            offset = (R @ local.unsqueeze(-1)).squeeze(-1)

        start = kept.numel() + clones.numel()
        values[0][start:] += offset
        values[1][start:] = (
            effective_log_s[split_parents] - math.log(split_scale)
        )

    #* These must be new leaf tensors because their shapes have changed.
    new_params = tuple(v.requires_grad_() for v in values)

    #* Preserve Adam moments for survivors; zero them for all new Gaussians.
    for old, new in zip(params, new_params):
        old_state = opt.state.pop(old, {})
        if not old_state:
            continue

        new_state = {}
        for key, value in old_state.items():
            if key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                moments = torch.zeros_like(new)
                moments[:kept.numel()] = value[kept]
                new_state[key] = moments
            else:
                # Includes Adam's scalar step counter.
                new_state[key] = (
                    value.clone() if torch.is_tensor(value) else value
                )

        opt.state[new] = new_state

    opt.param_groups[0]["params"] = list(new_params)
    return new_params

def train(N, H, W, target, budget, do_densify = True):
    dev = target.device

    # parameters (leaf tensors, requires_grad=True); a spread-out init, e.g.:
    mu     = (torch.rand(N, 2, device=dev) * torch.tensor([W, H], device=dev)).requires_grad_()         # (N, 2)  spread across the image
    log_s  = torch.full((N, 2), math.log(0.02 * max(H, W)), device=dev, requires_grad=True)   # (N, 2)  small blobs, log space
    theta  = torch.zeros(N, device=dev, requires_grad=True)                                   # (N,)    rotation
    color  = torch.zeros(N, 3, device=dev, requires_grad=True)                                # (N, 3)  sigmoid -> 0.5 gray
    op_raw = torch.full((N,), -2.0, device=dev, requires_grad=True)                           # (N,)    sigmoid -> ~0.12 opacity
    order = torch.arange(N, device=dev)

    opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)
    grad_sum = torch.zeros(N, device=dev)
    grad_steps = 0
    densify_every = 200
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
        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Accumulate magnitudes, not gradient vectors:
        # opposite gradient directions should not cancel.
        with torch.no_grad():
            grad_sum.add_(mu.grad.norm(dim=-1))
        grad_steps += 1

        opt.step()

        # Leave the final step for optimization rather than creating untrained children.
        if (step + 1) % densify_every == 0 and step + 1 < 2000 and do_densify:
            mu, log_s, theta, color, op_raw = densify(
                (mu, log_s, theta, color, op_raw),
                opt,
                grad_sum / grad_steps,
                budget,
                size_threshold=0.02 * W,
                max_log_scale=math.log(max(H, W) / 8),
            )

            order = torch.arange(mu.shape[0], device=dev)
            grad_sum = torch.zeros(mu.shape[0], device=dev)
            grad_steps = 0
        # psnr = -10 * torch.log10(loss)
        torch.mps.synchronize()
        end_time = time.perf_counter()

        #print(f"Step spent {end_time - start_time} seconds!")
        

    save_image(finalImg, f"results/final-coffee-{N}.png")

def train3d(N, train_cameras, iters, dev, budget, do_densify=True):
    # parameters (leaf tensors, requires_grad=True); example init for this scene:
    mu3    = ((torch.rand(N, 3, device=dev) * 2 - 1) * 1.5).requires_grad_()     # (N, 3)  cloud in ~[-1.5, 1.5]^3
    log_s  = torch.log(0.08 * torch.ones(N, 3, device=dev)).requires_grad_()   # (N, 3)  small 3D blobs
    quat   = torch.zeros(N, 4, device=dev); quat[:, 0] = 1.0; quat.requires_grad_()     # (N, 4)  identity rotation (w, x, y, z)
    color  = torch.zeros(N, 3, device=dev, requires_grad=True)                       # (N, 3)  sigmoid -> gray
    op_raw = torch.full((N,), -2.0, device=dev, requires_grad=True)                  # (N,)    sigmoid -> low opacity
    opt    = torch.optim.Adam([mu3, log_s, quat, color, op_raw], lr=1e-2)
    grad_sum = torch.zeros(N, device=dev)
    grad_steps = 0
    densify_every = 200
    finalImg = None

    for step in range(iters):                        # e.g. N = 4000 Gaussians, iters = 1500
        cam = random_choice(train_cameras, dev)
        img = render3d(mu3, log_s, quat, color, op_raw, cam)

        finalImg = img
        loss  = ((img - cam.image) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()

        if step % 100 == 0:
            print(step, "K", render.K, loss)

        with torch.no_grad():
            grad_sum.add_(mu3.grad.norm(dim=-1))
        grad_steps += 1

        opt.step()

        #* opacity reset (not part of writeup but standard for gaussians +
        #* leads to better results in this case)
        if step + 1 == 1000:
            with torch.no_grad():
                op_raw.clamp_(max=math.log(0.01 / 0.99))
                opt.state[op_raw]["exp_avg"].zero_()
                opt.state[op_raw]["exp_avg_sq"].zero_()

        #* densify step
        if (step + 1) % densify_every == 0 and step + 1 < iters and do_densify:
            mu3, log_s, quat, color, op_raw = densify(
                (mu3, log_s, quat, color, op_raw),
                opt,
                grad_sum / grad_steps,
                budget,
                size_threshold=0.05,  # Starting value in world units; tune as needed.
            )

            grad_sum = torch.zeros(mu3.shape[0], device=dev)
            grad_steps = 0

    save_image(finalImg, f"results/final-spheres-{budget}-{N}-{iters}.png")