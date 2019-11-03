import vis.visutils as vis

def log_losses_tb(opt, writer, losses, global_stepsize):
    disc_loss, gen_loss, task_loss = losses
    writer.add_scalar('Generator Loss', gen_loss, global_stepsize)
    writer.add_scalar('Discriminator Loss', disc_loss, global_stepsize)
    writer.add_scalar('Classifier Loss', task_loss, global_stepsize)

def log_comparisons_grid(opt, writer, real, fake):
    writer.add_image('Generated Samples', vis.get_img_grid(real, fake, n=opt.log_n_images))

def log_predictions_grid(opt, writer, classifier, images, global_stepsize):
    writer.add_figure('predictions on target domain',
                             vis.plot_classes_preds(classifier, images.detach()[:opt.log_n_images]),
                             global_stepsize)