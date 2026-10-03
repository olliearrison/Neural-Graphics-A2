import torch
import numpy as np

#* idk if this should be in helpers



def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"