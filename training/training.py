import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from models.gan import get_modules
import utils
from datasets import dataloaders
from vis.visutils import tb_writer, select_n_random
import vis.tblog as tblog

def train_gan(opt):
    print(f'Training for {opt.n_epochs} epochs...\n')
    FloatTensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if opt.cuda else torch.LongTensor

    # Get GAN Modules
    modules = get_modules(opt)

    discriminator = modules['Discriminator']
    generator = modules['Generator']
    classifier = modules['Classifier']

    discriminator.apply(utils.weights_init_normal)
    generator.apply(utils.weights_init_normal)

    adversarial_loss = nn.BCELoss()
    task_loss = nn.CrossEntropyLoss()

    optim_G = optim.Adam(itertools.chain(generator.parameters(), classifier.parameters()), lr=lr, weight_decay=l2_decay, betas=(opt_beta_1, opt_beta_2))
    optim_D = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=l2_decay, betas=(opt_beta_1, opt_beta_2))
    optim_C = optim.Adam(classifier.parameters(), lr=lr, weight_decay=l2_decay, betas=(opt_beta_1, opt_beta_2))

    scheduler_G = optim.lr_scheduler.StepLR(optim_G, step_size=decay_every, gamma=lr_decay)
    scheduler_D = optim.lr_scheduler.StepLR(optim_D, step_size=decay_every, gamma=lr_decay)
    scheduler_C = optim.lr_scheduler.StepLR(optim_C, step_size=decay_every, gamma=lr_decay)

    source_loader, target_loader = dataloaders.get_dataloaders(opt)

    writer = tb_writer(opt.tensorboard_logs)

    for e in range(opt.n_epochs):

        running_gen_loss = 0.0
        running_disc_loss = 0.0
        running_task_loss = 0.0

        for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(source_loader, target_loader)):
            
            N = imgs_A.shape[0]

            imgs_A = imgs_A.type(FloatTensor).expand(N, 3, img_size, img_size)
            labels_A = labels_A.type(LongTensor)
            imgs_B = imgs_B.type(FloatTensor)
            
            truth_label = FloatTensor(N).fill_(1.0)
            false_label = FloatTensor(N).fill_(0.0)

            # --------------
            # Train Generator
            # --------------
            
            noise_z = FloatTensor(np.random.uniform(-1, 1, (batch_size, latent_dim)))
            
            optim_G.zero_grad()
            
            fake_img = generator(imgs_A, noise_z)
            pred = classifier(fake_img)
            lt = (task_loss(pred, labels_A) + task_loss(classifier(imgs_A), labels_A))/2
            gen_loss = gen_loss_weight*adversarial_loss(discriminator(fake_img), truth_label) + classifier_loss_weight*lt
            gen_loss.backward()
            optim_G.step()
            scheduler_G.step()

            running_gen_loss += gen_loss.item()
            running_task_loss += classifier_loss_weight*lt.item()

            # --------------
            # Train Discriminator
            # --------------
            
            optim_D.zero_grad()
            
            ld = adversarial_loss(discriminator(fake_img.detach()), false_label) + adversarial_loss(discriminator(imgs_B), truth_label)
            disc_loss = ld*disc_loss_weight
            disc_loss.backward()
            optim_D.step()
            scheduler_D.step()
            
            running_disc_loss += disc_loss.item()

            losses = [disc_loss.item(), gen_loss.item(), classifier_loss_weight*lt.item()]

            # --------------
            # Logging
            # --------------
            if i % opt.log_every == opt.log_every-1:
                global_stepsize = e*len(source_loader)+i
                sample_real, sample_fake = select_n_random(imgs_A.detach(), fake_img.detach())

                tblog.log_losses_tb(opt, writer, losses, global_stepsize)
                tblog.log_comparison_grid(opt, writer, sample_real, sample_fake)
                tblog.log_predictions_grid(opt, writer, classifier, imgs_B)

                writer.close()
            
        
        print(f"[ Epoch #{e+1}/{opt.n_epochs} ]\t[ Batch #{i}/{len(source_loader)} ]\t[ Gen Loss: {running_gen_loss/N} ]\t[ Disc Loss: {running_disc_loss/N} ]\t[ Classifier Loss: {running_task_loss/N} ]")
        running_gen_loss = 0.0
        running_disc_loss = 0.0
        running_task_loss = 0.0