import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--epochs', type=int, default=20000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--decay_every', type=int, default=20000, help='decay interval for learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='decay rate for learning rate')
    parser.add_argument('--dropout_prob', type=float, default=0.9, help='dropout retention rate for discriminator')
    parser.add_argument('--lrelu_slope', type=float, default=0.2, help='slope of leaky relu function')
    parser.add_argument('--noise_mu', type=float, default=0.0, help='mean of gaussian noise vector')
    parser.add_argument('--noise_std', type=float, default=0.2, help='standard deviation of gaussian noise vector')
    parser.add_argument('--weight_init_mu', type=float, default=0.0, help='mean of network weights from normal distribution')
    parser.add_argument('--weight_init_std', type=float, default=0.2, help='standard deviation of network weights from normal distribution')
    parser.add_argument('--opt_beta_1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--opt_beta_2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--n_resblocks', type=int, default=6, help='number of residual blocks in generator')
    parser.add_argument('--img_size', type=int, default=32, help='size of images')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--channels', type=int, default=3, help='number of channels in images')
    parser.add_argument('--latent_dim', type=int, default=10, help='latent dimension of noise vector')
    parser.add_argument('--disc_loss_weight', type=float, default=0.13, help='loss weight for discriminator')
    parser.add_argument('--gen_loss_weight', type=float, default=0.011, help='loss weight for generator')
    parser.add_argument('--classifier_loss_weight', type=float, default=0.01, help='loss weight for classifier')
    parser.add_argument('--data_download_dir', type=str, default='data', help='directory to download datasets')
    return parser

if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()

    print('Learning rate is: ', opt.lr)