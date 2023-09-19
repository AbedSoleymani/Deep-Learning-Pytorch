import os
import torch
import matplotlib.pyplot as plt

from siren import Siren
from mlp import MLP
from train_model import train_model
from input import gen_input
from plot_results import plot_results
import device

os.system("clear")

n_epochs = 200

siren = Siren().to(device.device)
mlp = MLP().to(device.device)

img, pixel_values, pixel_coordinates, resolution = gen_input(default=False)

for i, net in enumerate([mlp, siren]):
    
    print('ReLU training' if (i == 0) else 'SIREN training')
    optim = torch.optim.Adam(lr=1e-3, params=net.parameters()) # set lr=1e-4 for smoother learning

    psnr, model_output = train_model(model=net,
                                     model_optimizer=optim,
                                     pixel_coordinates=pixel_coordinates,
                                     pixel_values=pixel_values,
                                     n_epochs=n_epochs)
    if i == 0:
        mlp_psnr, mlp_output = psnr, model_output
    else:
        siren_psnr, siren_output= psnr, model_output

plot_results(img,
             mlp_psnr, mlp_output,
             siren_psnr, siren_output,
             resolution)