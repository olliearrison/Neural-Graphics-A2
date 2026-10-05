from rendering import *
from training import *
from utils import *


def main():
    targetImg = image_to_tensor("data/targets/cat.png")
    targetImg.permute(1, 2, 0)
    H, W = targetImg.shape[0], targetImg.shape[1]

    print(targetImg.shape)

    targetImg.to(device=get_device())

    train(N=256, H=H, W=W, target=targetImg)


main()