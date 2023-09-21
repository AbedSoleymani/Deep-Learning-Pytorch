import numpy as np
from tqdm import tqdm

def train_model(model,
                timestamp,
                values,
                model_optimizer,
                n_epochs=100):
    model.train()
    psnr = []
    for _ in tqdm(range(n_epochs)):
        model_output = model(timestamp)
        loss = ((model_output - values) ** 2).mean()
        psnr.append(20 * np.log10(1.0 / np.sqrt(loss.item())))

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

    return psnr, model_output