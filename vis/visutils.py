import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img/2 + 0.5
    if img.is_cuda:
        npimg = img.cpu().numpy()
    else:
        npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))
        

def select_n_random(truth, gen, n=4):
    '''
    https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    Selects n random Xt and their corresponding Xf
    '''
    assert len(truth) == len(gen), 'Dataset sizes must be same in dim 0'
    perm = torch.randperm(len(truth))
    return truth[perm][:n], gen[perm][:n]


def get_img_grid(truth, gen, n=4):
    assert truth.shape == gen.shape, 'Shapes of truth and gen sets must be equal'
    if truth.is_cuda:
        truth = truth.cpu()
    if gen.is_cuda:
        gen = gen.cpu()
    imgs = torch.cat((truth, gen), dim=0)
    return make_grid(imgs, nrow=n)
    
def images_to_probs(classifier, images):
    '''
    https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = classifier(images)
    _, preds = torch.max(output, 1)
    if preds.is_cuda:
        preds = np.squeeze(preds.cpu().numpy())
    else:
        preds = np.squeeze(preds.numpy())
    return preds, [F.softmax(el)[i].item() for i,el in zip(preds, output)]

def plot_classes_preds(classifier, images):
    '''
    https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(classifier, images)
    fig = plt.figure(figsize=(6, 12))
    for idx in range(len(images)):
        ax = fig.add_subplot(1,4,idx+1,xticks=[],yticks=[])
        matplotlib_imshow(images[idx])
        ax.set_title("{0: .1f}%\npredicted: {1}".format(
                    probs[idx]*100,
                    preds[idx]))
    return fig

def tb_writer(logdir):
    return SummaryWriter(logdir)