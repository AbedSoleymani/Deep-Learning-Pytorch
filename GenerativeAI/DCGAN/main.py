import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator, Generator, initialize_weights
from save_png import save_png
from save_gif import save_gif

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

lr = 1e-3
batch_size = 128
img_size = 64
channels_img = 1 # for mnist
z_dim = 100
epochs = 1
features_disc = 64
features_gen = 64

transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * channels_img, [0.5] * channels_img)
])

dataset = datasets.MNIST(root= "./GenerativeAI/DCGAN/dataset/", train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
gen = Generator(z_dim, channels_img, features_gen).to(device)
disc = Discriminator(channels_img, features_disc).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

save_path = "./GenerativeAI/DCGAN/logs/fake"
os.makedirs(save_path, exist_ok=True)

step = 0

for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(loader):
        gen.train()
        disc.train()
        real = real.to(device)
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        fake = gen(noise)
        
        # Train Discriminator: max{Log[D(x)] + Log[1 - D(G(Z))]}
        disc_real = disc(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        disc.zero_grad()
        """ `retain_graph=True` in the next line of code:
            Retains the computational graph for subsequent backward passes
        """
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator: min{Log[1 - D(G(Z))]} or equivalently max{Log[D(G(Z))]}
        disc_fake = disc(fake).view(-1)
        loss_gen = criterion(disc_fake, torch.ones_like(disc_fake))

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(loader)} "
                f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
            )

            gen.eval()
            with torch.no_grad():
                fake = gen(fixed_noise)
                # Take out first 32 examples
                img_grid_fake = torchvision.utils.make_grid(fake[:32,0,:,:], normalize=True)
                file_path = f"{save_path}/{str(step)}.png"
                im = img_grid_fake.cpu().numpy()
                save_png(image=im, file_path=file_path)

            step += 1

save_gif(step=step)