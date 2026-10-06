from rendering import *
from training import *
from utils import *
import json

def main():
    # targetImg = image_to_tensor("data/targets/coffee.png")
    # targetImg.permute(1, 2, 0)
    # H, W = targetImg.shape[0], targetImg.shape[1]

    # print(targetImg.shape)

    # targetImg.to(device=get_device())

    # train(N=4096, H=H, W=W, target=targetImg)


    camera_json = json.load(open("data/spheres/cameras.json"))

    train3d(
        N=1024,
        train_cameras=camera_json,
        iters=2000,
        dev=get_device(),
        budget=4096,
        do_densify=True
    )


main()