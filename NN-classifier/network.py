import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from data_loader import MNISTLoader

class Network(nn.Module):
    def __init__(self, num_inputs=784, num_hidden=[128,64], num_outputs=10):
        super(Network, self).__init__() # calls the __init__ method in nn.Module
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        self.fc1 = nn.Linear(self.num_inputs, self.num_hidden[0])
        self.fc2 = nn.Linear(self.num_hidden[0], self.num_hidden[1])
        self.fc3 = nn.Linear(self.num_hidden[1], self.num_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x) # until here, x has the dimension on batch_size*num_outputs
        x = F.softmax(x, dim=1) # dim=1 specifies the second dimesion (num_outputs) where the softmax is applied

        return x
    
    def train(self, epochs = 50, batch_size=128, lr = 0.01):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        mnist_data = MNISTLoader(batch_size=batch_size)
        data_generator, _, _ = mnist_data.train_set_generator()

        print_every = 50
        steps = 0
        for e in range(epochs):
            running_loss = 0
            for images, labels in data_generator:
                steps += 1
                images.resize_(batch_size, 784)

                optimizer.zero_grad()

                output = self.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    print("Epoch: {}/{},  ".format(e+1, epochs),
                          "Loss: {:.4f}".format(running_loss/print_every))
                    running_loss = 0
