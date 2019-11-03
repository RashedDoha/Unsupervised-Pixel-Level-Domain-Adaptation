def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, weight_init_mu, weight_init_std)
    elif type(m) == nn.BatchNorm2d or type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, weight_init_mu, weight_init_std)
        nn.init.constant_(m.bias.data, 0.0)