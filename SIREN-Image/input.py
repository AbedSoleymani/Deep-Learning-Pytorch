import torch
import cv2
import skimage
import numpy as np

import device

def gen_input(default=True):
    if default:
        img = ((torch.from_numpy(skimage.data.camera()) - 127.5) / 127.5) # hxw matrix
    else:
        img = np.asarray(cv2.imread('./SIREN-Image/imgs/Abed.png', cv2.IMREAD_GRAYSCALE))
        img = cv2.resize(img, (500, 500))
        img = ((torch.from_numpy(img) - 127.5) / 127.5)

    pixel_values = img.reshape(-1, 1).to(device.device) # lx1 vector: like a timeseries

    # Input
    resolution = img.shape[0]
    tmp = torch.linspace(-1, 1, steps=resolution)
    x, y = torch.meshgrid(tmp, tmp)
    pixel_coordinates = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).to(device.device)

    return img, pixel_values, pixel_coordinates, resolution