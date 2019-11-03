from config import *
from datasets import dataloaders
import torch
from training import training
def main():
    parser = get_arguments()
    opt = parser.parse_args()
    if opt.cuda is False and torch.cuda.is_available():
        print('You have a CUDA enabled GPU! You can turn on GPU accelerated training with the flag --cuda')
    training.train_gan(opt)


if __name__ == '__main__':
    main()