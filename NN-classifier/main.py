from data_loader import MNISTLoader
from network import Network
from helper import view_classify


my_mnist = MNISTLoader(batch_size=128, train=True)
_, images, labels = my_mnist.train_set_generator()

model = Network()
images.resize_(my_mnist.batch_size, 1, model.num_inputs)

ps = model.forward(images[0])
view_classify(images[0], ps, title="before training")

model.train(epochs=5, batch_size=96)

ps = model.forward(images[0])
view_classify(images[0], ps, title="after training")
