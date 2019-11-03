import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, opt, in_features=3, out_features=64):
        super(Discriminator, self).__init__()
        def block(in_features, out_features):
            layers = [
                nn.Conv2d(in_features, out_features, 3, 2, 1),
                nn.Dropout(p=1-opt.dropout_prob),
                nn.BatchNorm2d(out_features),
                nn.LeakyReLU(negative_slope=opt.lrelu_slope, inplace=True)
            ]
            
            return layers
        
        self.l1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.Dropout(p=1-opt.dropout_prob),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=opt.lrelu_slope, inplace=True)
        )
        
        self.blocks = nn.Sequential(
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024)
        )
        
        self.l2 = nn.Sequential(
            nn.Linear(1024*(int(opt.img_size/2**4))**2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        x = self.l1(img)
        x = self.blocks(x)
        out = self.l2(x.view(img.shape[0], -1))
        return out