import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        def block(in_features, out_features, parameterized=True):
            if parameterized:
                layers = [nn.Conv2d(in_features, out_features, 5, 1), nn.PReLU(), nn.MaxPool2d(2, stride=2)]
            else:
                layers = [nn.Conv2d(in_features, out_features, 5, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2)]
            return layers
        
        reduced_img_size = ((opt.img_size - 4)/2 - 4)/2
        
        self.private = nn.Sequential(
            *block(channels, 32)
        )
        self.shared = nn.Sequential(
            *block(32, 48),
            Flatten(),
            nn.Linear(48*(reduced_img_size**2), 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )
        
    def forward(self, img):
        return self.shared(self.private(img))