import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, out_features, 3, 1, 1),
            nn.BatchNorm2d(out_features)
        )
        
    def forward(self, x):
        return x + self.block(x)