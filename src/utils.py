import math
from torch import Tensor



#* idk if this should be in helpers
def covariance_2d(scale, theta):
    # scale: (N, 2) positive,  theta: (N,) radians
    # TODO: build R(theta) and S = diag(scale), return Sigma = R S S^T R^T  -> (N, 2, 2)
    R = R(theta)
    S = diag(scale)
    return ...

def gaussian_weight(xy, mu, Sigma):
    # xy: (P, 2) pixel coords,  mu: (N, 2),  Sigma: (N, 2, 2)
    # TODO: w[p, n] = exp(-0.5 (xy_p - mu_n)^T Sigma_n^-1 (xy_p - mu_n))
    #                          # (P, N)
    N = mu.shape[0]
    P = xy.shape[0]
    result = Tensor.torch.zeros(N,P)

    for n in range(N):
        for p in range(P):
            d = xy[p] - mu[n]
            #* Documentation suggests using solve for numerical stability 
            #* should be equiv to A.inv() @ B
            #? https://docs.pytorch.org/docs/2.14/generated/torch.linalg.inv.html
            solved = Tensor.torch.linalg.solve(Sigma[n], d)
            dist_sq = d @ solved
            result[p, n] = Tensor.torch.exp(-0.5 * dist_sq)

    return result


#* image is a Tensor, path is where we want to save it
def save_image(image, path):
    return 42