from rendering import *
from training import *
from utils import *
import json

def main():
    # targetImg = image_to_tensor("data/targets/cat.png")
    # targetImg.permute(1, 2, 0)
    # H, W = targetImg.shape[0], targetImg.shape[1]

    # print(targetImg.shape)

    # targetImg.to(device=get_device())

    # train(N=4096, H=H, W=W, target=targetImg, budget=0, do_densify=False)


    camera_json = json.load(open("data/spheres/cameras.json"))

    PSNR_dict = dict()

    for index in range(12):
        final_PSNR=    train3d(
                            N=2048,
                            train_cameras=camera_json,
                            iters=2000,
                            dev=get_device(),
                            budget=4096,
                            do_densify=False,
                            do_validate=True,
                            index=index
                        )
        PSNR_dict[index] = final_PSNR

    print(PSNR_dict)

main()