import torch.nn as nn
from models.resblock import ResidualBlock

class Generator(nn.Module):
    def __init__(self, opt, l1_features=64):
        super(Generator, self).__init__()
        self.fc = nn.Linear(opt.latent_dim, opt.channels*opt.img_size**2)
        self.l1 = nn.Sequential(nn.Conv2d(opt.channels*2, l1_features, 3, 1, 1), nn.ReLU(inplace=True))
        
        resblocks = []
        for _ in range(opt.n_resblocks):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)
        
        self.l2 = nn.Sequential(nn.Conv2d(l1_features, opt.channels, 3, 1, 1), nn.Tanh())
        
    def forward(self, img, z):
        x = torch.cat((img, self.fc(z).view(*opt.img.shape)), dim=1)
        x = self.l1(x)
        x = self.resblocks(x)
        x = self.l2(x)
        return x