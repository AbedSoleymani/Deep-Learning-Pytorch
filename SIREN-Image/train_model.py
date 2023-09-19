import numpy as np
from tqdm import tqdm

def train_model(model,
                pixel_coordinates,
                pixel_values,
                model_optimizer,
                n_epochs=100):
    model.train()
    psnr = []
    for _ in tqdm(range(n_epochs)):
        model_output = model(pixel_coordinates)
        loss = ((model_output - pixel_values) ** 2).mean()
        psnr.append(20 * np.log10(1.0 / np.sqrt(loss.item())))

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

    return psnr, model_output