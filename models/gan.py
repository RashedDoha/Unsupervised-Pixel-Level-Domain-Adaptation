from models.discriminator import Discriminator
from models.generator import Generator
from models.classifier import Classifier

def get_modules(opt):
    modules = {}
    disc = Discriminator()
    gen = Generator()
    clf = Classifier()
    if opt.cuda:
        disc = disc.cuda()
        gen = gen.cuda()
        clf = clf.cuda()
    
    modules['Discriminator'] = disc
    modules['Generator'] = gen
    modules['Classifier'] = clf
    return modules