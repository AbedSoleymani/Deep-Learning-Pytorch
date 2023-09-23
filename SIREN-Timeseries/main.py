import os
import torch
import matplotlib.pyplot as plt

from siren import Siren
from mlp import MLP
from train_model import train_model
from input import gen_timeseries
from plot_results import plot_results
import device

os.system("clear")

n_epochs = 2000

siren = Siren(in_dim=1).to(device.device)
mlp = MLP(in_dim=1).to(device.device)

timestamp, values = gen_timeseries(noise=False)

for i, net in enumerate([mlp, siren]):
    
    print('ReLU training' if (i == 0) else 'SIREN training')
    optim = torch.optim.Adam(lr=1e-5, params=net.parameters()) # set lr=1e-4 for smoother learning

    psnr, model_output = train_model(model=net,
                                     model_optimizer=optim,
                                     timestamp=timestamp,
                                     values=values,
                                     n_epochs=n_epochs)
    if i == 0:
        mlp_psnr, mlp_output = psnr, model_output
    else:
        siren_psnr, siren_output= psnr, model_output

plot_results(values,
             mlp_output,
             siren_output,
             n_epochs)
