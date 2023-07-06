from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

class MNISTLoader:
    def __init__(self, batch_size=64, shuffle=True, train=True, path='MNIST_data/'):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train
        self.path = path

    def transform(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,),(0.5,)),
                                        ])
        return transform
    
    def train_set_generator(self):
        train_set = datasets.MNIST(self.path, download=True, train=self.train, transform=self.transform())
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=self.shuffle)
        # train_loader is a generator that can be iterated by 'iter'.
        train_iter = iter(train_loader)
        # We can recieve the next batch by using the 'next' command.
        images, labels = next(train_iter)
        return train_loader, images, labels
    
    def show_rand_img(self, images):
        img = images[random.randrange(self.batch_size)].numpy().squeeze()
        plt.imshow(img, cmap='Grays_r')
        plt.show()