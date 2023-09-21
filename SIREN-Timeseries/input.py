import torch
import numpy as np
import matplotlib.pyplot as plt

def gen_timeseries(num_timestamps=500, noise=False, plot=False):

    t = np.linspace(0, 1, num_timestamps)
    timeseries = np.cos(np.pi * t) + 2 * np.sin(2 * np.pi * t) + 1.5 * np.cos(5 * np.pi * t)

    if noise:
        n = np.random.normal(0, 0.1, num_timestamps)
        timeseries += n

    timeseries /= np.max(np.abs(timeseries))

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(t, timeseries, label="Smooth Time Series")
        plt.title("Random Smooth Time Series")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    t = torch.from_numpy(t).float()
    # Reshape the tensor to match the expected input shape (batch_size, 1)
    t = t.view(-1, 1)

    timeseries = torch.from_numpy(timeseries).float()
    timeseries = timeseries.reshape(-1, 1)
    
    return t, timeseries