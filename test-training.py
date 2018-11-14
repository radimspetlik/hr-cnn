'''Pytorch training procedure of the SNR extractor with CrossEntropy loss

Usage:
  %(prog)s  [--net-architecture=<string>]
            [--batch-size=<int>] [--test-batch-size=<int>] [--epochs=<int>]
            [--lr=<float>] [--momentum=<float>] [--no-cuda]
            [--plot-after=<int>]
            [--x-lmdb-path-train=<path>]
            [--y-lmdb-path-train=<path>]
            [--x-lmdb-path-validation=<path>]
            [--y-lmdb-path-validation=<path>]
            [--output-to=<path>] [--continue-model=<path>]
            [--plots-path=<path>]
            [--seed=<int>]

  %(prog)s (--help | -h)

Options:
    --net-architecture=<string>        net architecture to test
    --batch-size=<int>                 input batch size for training [default: 64]
    --test-batch-size=<int>            input batch size for testing [default: 128]
    --epochs=<int>                     number of epochs to train [default: 10]
    --lr=<float>                       learning rate [default: 0.01]
    --momentum=<float>                 SGD momentum [default: 0.5]
    --seed=<int>                       random seed [default: 1]
    --plot-after=<int>                 number of epochs after which the test signal with estimation will be plotted [default: 500]
    --x-lmdb-path-train=<path>         X LMDB training DB
    --y-lmdb-path-train=<path>         y LMDB training DB
    --x-lmdb-path-validation=<path>    X LMDB validation DB
    --y-lmdb-path-validation=<path>    y LMDB validation DB
    --output-to=<path>                    absolute path to the directory where the model should be stored
    --continue-model=<path>            absolute path to the directory where the model with which the learning should be continued
    --plots=<path>                     absolute path to the directory where the plots should be stored
    --no-cuda                          disables CUDA training
    -h, --help                             should be help but none is given

See '%(prog)s --help' for more information.

'''

from __future__ import print_function
from PIL import Image
from cmp.nrppg.torch.TorchLossComputer import TorchLossComputer
import sys
import logging
from docopt import docopt
import torch
import torch.optim as optim
from torch.autograd import Variable
import datetime
import random
import time
import os.path
import numpy as np
from torchvision import transforms
import socket


__logging_format__ = '[%(levelname)s]%(message)s'
logging.basicConfig(format=__logging_format__, stream=sys.stdout)
logger = logging.getLogger("train_log")
logging.getLogger("train_log").setLevel(logging.INFO)

prog = os.path.basename(sys.argv[0])
completions = dict(
    prog=prog,
)
args = docopt(
    __doc__ % completions,
    argv=sys.argv[1:],
    version='SNR estimator',
)
cuda = not bool(args['--no-cuda']) and torch.cuda.is_available()

torch.manual_seed(int(args['--seed']))
if cuda:
    torch.cuda.manual_seed(int(args['--seed']))


def train(model_to_train, optimizer, training_epoch):
    start = time.time()

    model_to_train.train()
    train_rmses = []
    train_losses = []
    train_aes = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model_to_train(data)

        # if training_epoch > 0 and training_epoch % int(args['--plot-after']) == 0:
        #     store_snrs_to_summary(val_ds, batch_idx, output, target, training_epoch)

        Fs, regularization_factor = train_ds.get_fps_and_regularization_factor(batch_idx * int(args['--batch-size']))
        train_loss, train_rmse, train_ae = TorchLossComputer.cross_entropy_power_spectrum_loss(output, target, Fs, regularization_factor,
                                                                                     cuda=True)

        if len(train_rmses) == 0:
            train_rmses = train_rmse
            train_losses = train_loss
            train_aes = train_ae
        else:
            train_rmses = torch.cat((train_rmses, train_rmse), dim=0)
            train_losses = torch.cat((train_losses, train_loss), dim=0)
            train_aes = torch.cat((train_aes, train_ae), dim=0)

        train_loss.backward()
        optimizer.step()

    end = time.time()

    return train_losses.data.cpu().numpy(), train_rmses.data.cpu().numpy(), train_aes.data.cpu().numpy(), end - start


def validate(model_to_validate, training_epoch):
    model_to_validate.eval()
    validation_rmses = []
    validation_aes = []
    validation_losses = []
    for batch_idx, (data, target) in enumerate(validation_loader):
        if cuda:
            data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        output = model_to_validate(data)

        Fs, regularization_factor = train_ds.get_fps_and_regularization_factor(batch_idx * int(args['--batch-size']))
        validation_loss, validation_rmse, validation_ae = TorchLossComputer.cross_entropy_power_spectrum_loss(output,
                                                                                                target, Fs, regularization_factor,
                                                                                                cuda=True)
        if len(validation_rmses) == 0:
            validation_rmses = validation_rmse
            validation_losses = validation_loss
            validation_aes = validation_ae
        else:
            validation_rmses = torch.cat((validation_rmses, validation_rmse), dim=0)
            validation_losses = torch.cat((validation_losses, validation_loss), dim=0)
            validation_aes = torch.cat((validation_aes, validation_ae), dim=0)

    return validation_losses.data.cpu().numpy(), validation_rmses.data.cpu().numpy(), validation_aes.data.cpu().numpy()


def load_model_prepare_losses_file():
    from cmp.nrppg.cnn.ModelLoader import ModelLoader

    epoch_shift = 0
    model_name = str(datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S-%f')) + '_' + basename

    # dynamically initialize net architecture
    if args['--continue-model'] is not None:
        # change model name
        model_name = args['--continue-model'].split('/')[-1]
        epoch_string = model_name.split('_')[-1].split('=')[-1]
        epoch_shift = int(epoch_string) + 1
        model_name = '_'.join(model_name.split('_')[:-1])

        from cmp.nrppg.cnn.ModelLoader import ModelLoader

        model, rgb = ModelLoader.load_model(args['--continue-model'])
    else:
        # logger.info(args['--net-architecture'])
        model, rgb = ModelLoader.initialize_model(args['--net-architecture'], model_type='extractor')

    if cuda:
        model.cuda()
    losses_filepath = args['--output-to'] + model_name + '_losses.npy'
    losses = np.zeros((int(args['--epochs']) + 1, 12))
    if os.path.isfile(losses_filepath):
        losses = np.load(losses_filepath)
        if losses.shape[0] != int(args['--epochs']) + 1:
            oldl = losses
            losses = np.zeros((int(args['--epochs']) + 1, 12))
            losses[:oldl.shape[0], :] = oldl

    return model_name, model, rgb, losses, epoch_shift


def prepare_loaders(rgb):
    trnsfm = None
    if socket.gethostname() == 'boruvka':
        trnsfm = transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.RandomCrop((150, 100)),
                                     transforms.RandomRotation(5, resample=Image.BILINEAR, expand=True),
                                     transforms.CenterCrop((150, 100)),
                                     transforms.Resize((192, 128), interpolation=Image.BILINEAR)])

    from cmp.nrppg.cnn.dataset.FaceDatasetLmdb import FaceDatasetLmdb
    train_ds = FaceDatasetLmdb(args['--x-lmdb-path-train'], args['--y-lmdb-path-train'], int(args['--batch-size']),
                               train=True, skip_partitioning=True, rgb=rgb, transform=trnsfm)
    val_ds = FaceDatasetLmdb(args['--x-lmdb-path-validation'], args['--y-lmdb-path-validation'], int(args['--batch-size']),
                             train=False, skip_partitioning=True, rgb=rgb)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(args['--batch-size']), shuffle=False, num_workers=12)
    validation_loader = torch.utils.data.DataLoader(val_ds, batch_size=int(args['--batch-size']), shuffle=False, num_workers=6)

    return train_loader, validation_loader, train_ds, val_ds


def evaluate_model(model, model_name, losses, epoch_shift):
    optimizer = optim.Adam(model.parameters(), lr=float(args['--lr']), weight_decay=0.000)

    logger.info(' Learning with the model %s...' % model_name)
    validate(model, epoch_shift)
    for epoch in range(epoch_shift, int(args['--epochs']) + 1):
        trn_losses, trn_RMSEs, trn_AEs, training_time = train(model, optimizer, epoch)
        val_losses, val_RMSEs, val_AEs = validate(model, epoch)

        logger.info(
            '[%04d][TRN] %.6f MAE: %.1f MSE: %.1f, %.1f (%.0fs)' % (epoch, trn_losses.mean(), trn_AEs.mean(), trn_RMSEs.mean(), trn_RMSEs.max(), training_time))
        logger.info('[%04d][VAL] %.6f MAE: %.1f MSE: %.1f, %.1f' % (epoch, val_losses.mean(), val_AEs.mean(), val_RMSEs.mean(), val_RMSEs.max()))

        losses[epoch, :] = [trn_losses.mean(), np.median(trn_losses),
                            val_losses.mean(), np.median(val_losses),
                            trn_RMSEs.mean(), np.median(trn_RMSEs), trn_RMSEs.max(),
                            val_RMSEs.mean(), np.median(val_RMSEs), val_RMSEs.max(),
                            trn_AEs.mean(), val_AEs.mean()]

        if epoch > 0:
            if losses[epoch, 10] < losses[:epoch, 10].min():
                torch.save(model.state_dict(), os.path.join(args['--output-to'], model_name + '_epoch=%d_val_mae_best' % epoch))
            if losses[epoch, 2] < losses[:epoch, 2].min():
                torch.save(model.state_dict(),
                           os.path.join(args['--output-to'], model_name + '_epoch=%d_val_avg_loss_best' % epoch))

    return losses


if __name__ == '__main__':
    hr_directory = os.path.join('data')

    torch.manual_seed(0)
    random.seed(0)

    additional_info = '_ecg-fitness'
    basename = 'arch=%s_lr=%.0E_batch-size=%d%s' % (
        args['--net-architecture'], float(args['--lr']), int(args['--batch-size']), additional_info)

    logger.info(' %s' % basename)

    model_name, model, rgb, losses, epoch_shift = load_model_prepare_losses_file()
    train_loader, validation_loader, train_ds, val_ds = prepare_loaders(rgb)

    losses = evaluate_model(model, model_name, losses, epoch_shift)

    logger.info('Succesfully finished...')
