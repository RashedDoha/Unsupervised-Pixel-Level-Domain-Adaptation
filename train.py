from config import *
from datasets import dataloaders
def main():
    parser = get_arguments()
    opt = parser.parse_args()
    if opt.cuda is False and torch.cuda.is_available():
        print('You have a CUDA enabled GPU! You can turn on GPU accelerated training with the flag --cuda')
    s_loader, t_loader = dataloaders.get_dataloaders(opt)


if __name__ == '__main__':
    main()