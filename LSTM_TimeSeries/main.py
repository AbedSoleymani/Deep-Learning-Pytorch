import numpy as np
import torch
import torch.nn as nn
from data_generator import Data_generator
from rnn import RNN

import matplotlib.pyplot as plt

data = np.loadtxt("./LSTM_TimeSeries/data.txt", dtype=float)[510:2500] #2500
min_val, max_val = np.min(data), np.max(data)
data = -1 + 2 * (data - min_val) / (max_val - min_val)
data = data[:, np.newaxis]
print(data.shape)

data_gen = Data_generator()
time_steps =np.linspace(0, data.shape[0], data.shape[0])

rnn_model = RNN(input_size=1,
                output_size=1, # just predicting the next time step
                hidden_dim=10,
                n_layers=2)

test_input = torch.Tensor(data).unsqueeze(0) # adds the batch_size of 1 as first dimension

output, hidden = rnn_model.forward(test_input, None)
print(output.shape)

data_gen.display(input=data,
                 prediction=output.detach().numpy(),
                 time_steps=time_steps,
                 title='RNN prediction before training!')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn_model.parameters(),
                             lr=0.01) 
rnn_model.train(criterion=criterion,
                optimizer=optimizer,
                training_data=data,
                epochs=60,
                print_every=30)