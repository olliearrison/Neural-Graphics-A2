import numpy as np
import torch

H = 100
W = 100
N = 50
log_s  = torch.log(0.02 * max(H, W) * torch.ones(N, 2))   # (N, 2)  small blobs, log space
theta  = torch.zeros(N)                                   # (N,)    rotation


def covariance_2d(scale, theta):
    # scale: (N, 2) positive,  theta: (N,) radians
    # TODO: build R(theta) and S = diag(scale), return Sigma = R S S^T R^T  -> (N, 2, 2)
    S = np.empty((scale.shape[0], 2, 2))
    S[:, 0, 0] = scale[:, 0]
    S[:, 0, 1] = 0
    S[:, 1, 0] = 0
    S[:, 1, 1] = scale[:, 1]
    
    cosVals = np.cos(theta)
    sinVals = np.sin(theta)

    R = np.empty((theta.shape[0], 2, 2))
    R[:, 0, 0] = cosVals
    R[:, 0, 1] = -sinVals
    R[:, 1, 0] = sinVals
    R[:, 1, 1] = cosVals

    return R @ S @ np.matrix_transpose(S) @ np.matrix_transpose(R)




def gaussian_weight(xy, mu, Sigma):
    # xy: (P, 2) pixel coords,  mu: (N, 2),  Sigma: (N, 2, 2)
    # TODO: w[p, n] = exp(-0.5 (xy_p - mu_n)^T Sigma_n^-1 (xy_p - mu_n))
    #                          # (P, N)
    N = mu.shape[0]
    P = xy.shape[0]
    result = xy.new_zeros((P, N))

    for n in range(N):
        for p in range(P):
            d = xy[p] - mu[n]
            #* Documentation suggests using solve for numerical stability 
            #* should be equiv to A.inv() @ B
            #? https://docs.pytorch.org/docs/2.14/generated/torch.linalg.inv.html
            solved = torch.linalg.solve(Sigma[n], d)
            dist_sq = d @ solved
            result[p, n] = torch.exp(-0.5 * dist_sq)

    return result


#* image is a Tensor, path is where we want to save it
def save_image(image, path):
    return 42