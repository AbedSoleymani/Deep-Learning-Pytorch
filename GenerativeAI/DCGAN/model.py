import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        """In the paper they did not use BN in the first layer of Discriminator
           and in the last layer of Generator. This is why we do not use _block
           rightaway!
        """
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x64
            nn.Conv2d(in_channels=channels_img,
                      out_channels=features_d,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(0.2),
            
            # Input: N x features_d x 32 x 32
            self._block(in_channels=features_d,
                        out_channels=features_d*2,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            
            # Input: N x features_d*2 x 16 x 16
            self._block(in_channels=features_d*2,
                        out_channels=features_d*4,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            
            # Input: N x features_d*4 x 8 x 8
            self._block(in_channels=features_d*4,
                        out_channels=features_d*8,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            
            # Input: N x features_d*8 x 4 x 4
            nn.Conv2d(in_channels=features_d*8,
                      out_channels=1,
                      kernel_size=4,
                      stride=2,
                      padding=0),
            # Output: N x 1 x 1 x 1

            nn.Sigmoid(),
        ),

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False, # Since we are using BN
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)
    

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N × z_dim x 1 x 1
            self._block(in_channels=z_dim,
                        out_channels=features_g*16,
                        kernel_size=4,
                        stride=1,
                        padding=0),

            # Input: N x features_g*16 × 4 x 4
            self._block(in_channels=features_g*16,
                        out_channels=features_g*8,
                        kernel_size=4,
                        stride=2,
                        padding=1),
        
            # Input: N x features_g*8 × 8 x 8
            self._block(in_channels=features_g*8,
                        out_channels=features_g*4,
                        kernel_size=4,
                        stride=2,
                        padding=1),
        
            # Input: N x features_g*4 × 16 x 16
            self._block(in_channels=features_g*4,
                        out_channels=features_g*2,
                        kernel_size=4,
                        stride=2,
                        padding=1),
        
            # Input: N x features_g*2 × 32 x 32
            nn.ConvTranspose2d(
                in_channels=features_g*2,
                out_channels=channels_img,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            # Output: N x channels_img × 64 x 64
            nn.Tanh(), # to map the output to [-1, 1]
        ),

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False, # Since we are using BN
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.gen(x)
    
"""initialize_weights function is based on instructions in the original paper"""
def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)