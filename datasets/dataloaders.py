import os
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.mnistm import MNISTM

def get_dataloaders(opt, dataset='mnist/mnistm'):
    source_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if(dataset == 'mnist/mnistm'):
        mnist_ds = datasets.MNIST(os.path.join('datasets', opt.data_download_dir), download=True, train=True, transform=source_transform)
        mnistm_ds = MNISTM(os.path.join('datasets', opt.data_download_dir), download=True, train=True, transform=target_transform)
        mnist_loader = DataLoader(mnist_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_workers, drop_last=True)
        mnistm_loader = DataLoader(mnistm_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_workers, drop_last=True)

        return mnist_loader, mnistm_loader