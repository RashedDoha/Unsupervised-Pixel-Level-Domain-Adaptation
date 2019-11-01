from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import MNISTM

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
        mnist_ds = datasets.MNIST('./datasets'+opt.data_download_dir, download=True, train=True, transform=mnist_transform)
        mnistm_ds = MNISTM('./datasets'+opt.data_download_dir, download=True, train=True, transform=mnistm_transform)
        mnist_loader = DataLoader(mnist_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_workers, drop_last=True)
        mnistm_loader = DataLoader(mnistm_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_workers, drop_last=True)

        return mnist_loader, mnistm_loader