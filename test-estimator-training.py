'''Pytorch training procedure of the HR Estimator

Usage:
  %(prog)s  [--extractor-net-architecture=<string>]
            [--estimator-net-architecture=<string>]
            [--extractor-model-path=<string>]
            [--estimator-model-path=<string>]
            [--batch-size=<int>] [--test-batch-size=<int>] [--epochs=<int>]
            [--extractor-lr=<float>]
            [--estimator-lr=<float>]
            [--momentum=<float>] [--no-cuda]
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
    --extractor-net-architecture=<string>        extractor net architecture to train
    --estimator-net-architecture=<string>        estimator net architecture to train
    --extractor-model-path=<string>             extractor net architecture to continue training
    --estimator-model-path=<string>             estimator net architecture to continue training
    --batch-size=<int>                 input batch size for training [default: 64]
    --test-batch-size=<int>            input batch size for testing [default: 128]
    --epochs=<int>                     number of epochs to train [default: 10]
    --extractor-lr=<float>                       learning rate [default: 0.01]
    --estimator-lr=<float>                       learning rate [default: 0.01]
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


__logging_format__ = '[%(levelname)s]%(message)s'
logging.basicConfig(format=__logging_format__, stream=sys.stdout)
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger(os.path.basename(__file__)).setLevel(logging.INFO)

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


def train(extractor_model, estimator_model, extractor_optimizer, estimator_optimizer, training_epoch):
    start = time.time()

    train_sq_errs = []
    train_abs_errs = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data), Variable(target)
        extractor_optimizer.zero_grad()
        estimator_optimizer.zero_grad()

        Fs, regularization_factor = train_ds.get_fps_and_regularization_factor(batch_idx * int(args['--batch-size']))
        regularization_factor = 1.0 / regularization_factor

        ext_output = extractor_model(data).squeeze().unsqueeze(dim=0).unsqueeze(dim=0)
        output = estimator_model(ext_output).squeeze()

        target = torch.median(target * 60.0).type(torch.FloatTensor).cuda()

        train_abs_err = torch.abs(output - target)
        train_sq_err = (output - target) ** 2

        if len(train_sq_errs) == 0:
            train_sq_errs = train_sq_err
            train_abs_errs = train_abs_err
        else:
            train_sq_errs = torch.cat((train_sq_errs, train_sq_err), dim=0)
            train_abs_errs = torch.cat((train_abs_errs, train_abs_err), dim=0)

        (regularization_factor * train_abs_err).backward()
        if training_epoch % 2 == 0:
            extractor_optimizer.step()
        else:
            estimator_optimizer.step()

    end = time.time()

    return train_sq_errs.data.cpu().numpy(), train_abs_errs.data.cpu().numpy(), end - start


def validate(extractor_model, estimator_model):
    extractor_model.eval()
    estimator_model.eval()

    validation_sq_errs = []
    validation_abs_errs = []
    for batch_idx, (data, target) in enumerate(validation_loader):
        if cuda:
            data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)

        output = extractor_model(data).squeeze().unsqueeze(dim=0).unsqueeze(dim=0)
        output = estimator_model(output).squeeze()

        target = torch.median(target * 60.0).type(torch.FloatTensor).cuda()

        validation_abs_err = torch.abs(output - target)
        validation_sq_err = (output - target) ** 2

        if len(validation_sq_errs) == 0:
            validation_sq_errs = validation_sq_err
            validation_abs_errs = validation_abs_err
        else:
            validation_sq_errs = torch.cat((validation_sq_errs, validation_sq_err), dim=0)
            validation_abs_errs = torch.cat((validation_abs_errs, validation_abs_err), dim=0)

    return validation_sq_errs.data.cpu().numpy(), validation_abs_errs.data.cpu().numpy()


def load_model_prepare_losses_file(extractor_model_path, estimator_model_path):
    from cmp.nrppg.cnn.ModelLoader import ModelLoader

    epoch_shift = 0
    model_name = str(datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S-%f')) + '_' + basename

    # dynamically initialize net architecture
    if extractor_model_path is not None:
        extractor_model, rgb = ModelLoader.load_model(extractor_model_path, model_type='extractor')
        estimator_model, nothing = ModelLoader.initialize_model(args['--estimator-net-architecture'], model_type='estimator')

        if "MonteCarlo" in args['--estimator-net-architecture']:
            mc_conf = torch.load(os.path.join('_'.join(estimator_model_path.split('_')[:7]) + '_monte-carlo-configuration'))
            try:
                estimator_model.setup(mc_conf['active_layers'], mc_conf['max_pool_kernel_size'], mc_conf['conv_kernel_size'],
                                      mc_conf['conv_filter_size'])
            except AttributeError as e:
                estimator_model.module.setup(mc_conf['active_layers'], mc_conf['max_pool_kernel_size'], mc_conf['conv_kernel_size'],
                                             mc_conf['conv_filter_size'])
        estimator_model = ModelLoader.load_parameters_into_model(estimator_model, estimator_model_path)
    else:
        extractor_model, rgb = ModelLoader.initialize_model(args['--extractor-net-architecture'], model_type='extractor')
        estimator_model, nothing = ModelLoader.initialize_model(args['--estimator-net-architecture'], model_type='estimator')

    if cuda:
        extractor_model.cuda()
        estimator_model.cuda()
    losses_filepath = args['--output-to'] + model_name + '_losses.npy'
    losses = np.zeros((int(args['--epochs']) + 1, 4))
    if os.path.isfile(losses_filepath):
        losses = np.load(losses_filepath)
        if losses.shape[0] != int(args['--epochs']) + 1:
            oldl = losses
            losses = np.zeros((int(args['--epochs']) + 1, 4))
            losses[:oldl.shape[0], :] = oldl

    return model_name, extractor_model, estimator_model, rgb, losses, epoch_shift


def prepare_loaders(rgb):
    trnsfm = None

    from cmp.nrppg.cnn.dataset.FaceDatasetLmdb import FaceDatasetLmdb
    train_ds = FaceDatasetLmdb(args['--x-lmdb-path-train'], args['--y-lmdb-path-train'], int(args['--batch-size']),
                               train=False, skip_partitioning=True, rgb=rgb, transform=trnsfm)
    val_ds = FaceDatasetLmdb(args['--x-lmdb-path-validation'], args['--y-lmdb-path-validation'], int(args['--batch-size']),
                             train=False, skip_partitioning=True, rgb=rgb)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(args['--batch-size']), shuffle=False, num_workers=12)
    validation_loader = torch.utils.data.DataLoader(val_ds, batch_size=int(args['--batch-size']), shuffle=False, num_workers=6)

    return train_loader, validation_loader, train_ds, val_ds


def evaluate_model(extractor_model, estimator_model, model_name, losses, epoch_shift):
    logger.info(' Extractor learning rate: %.15f, estimator learning rate: %.15f ' % (float(args['--extractor-lr']),
                                                                                      float(args['--estimator-lr'])))
    extractor_optimizer = optim.Adam(extractor_model.parameters(), lr=float(args['--extractor-lr']), weight_decay=0.000)
    estimator_optimizer = optim.Adam(estimator_model.parameters(), lr=float(args['--estimator-lr']), weight_decay=0.000)

    logger.info(' Learning with the model %s...' % model_name)
    val_SQ_ERRs, val_ABS_ERRs = validate(extractor_model, estimator_model)
    logger.info('[%04d][VAL] MAE: %.1f RMSE: %.1f' % (0, val_ABS_ERRs.mean(), np.sqrt(val_SQ_ERRs.mean())))

    train_exs = {'extractor': 0, 'estimator': 1}
    train_exs_id = 'extractor'
    for epoch in range(epoch_shift, int(args['--epochs']) + 1):
        logger.info('[%04d] Training %s' % (epoch, train_exs_id))
        trn_SQ_ERRs, trn_ABS_ERRs, training_time = train(extractor_model, estimator_model,
                                                         extractor_optimizer, estimator_optimizer, train_exs[train_exs_id])
        val_SQ_ERRs, val_ABS_ERRs = validate(extractor_model, estimator_model)

        trn_RMSE = np.sqrt(trn_SQ_ERRs.mean())
        val_RMSE = np.sqrt(val_SQ_ERRs.mean())

        logger.info('[%04d][TRN] MAE: %.1f RMSE: %.1f (%.0fs)' % (epoch, trn_ABS_ERRs.mean(), trn_RMSE, training_time))
        logger.info('[%04d][VAL] MAE: %.1f RMSE: %.1f' % (epoch, val_ABS_ERRs.mean(), val_RMSE))

        losses[epoch, :] = [trn_RMSE, val_RMSE,
                            trn_ABS_ERRs.mean(), val_ABS_ERRs.mean()]

        if epoch > 0:
            if losses[epoch, 3] < losses[:epoch, 3].min():
                if train_exs_id == 'extractor':
                    train_exs_id = 'estimator'
                else:
                    train_exs_id = 'extractor'
                torch.save(extractor_model.state_dict(), os.path.join(args['--output-to'], model_name + '_extractor_val_mae_best'))
                torch.save(estimator_model.state_dict(), os.path.join(args['--output-to'], model_name + '_estimator_val_mae_best'))
            if losses[epoch, 1] < losses[:epoch, 1].min():
                torch.save(extractor_model.state_dict(), os.path.join(args['--output-to'], model_name + '_extractor_val_rmse_best'))
                torch.save(estimator_model.state_dict(), os.path.join(args['--output-to'], model_name + '_estimator_val_rmse_best'))

    return losses


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    additional_info = ''
    basename = 'ex-arch=%s_es-arch=%s_ex-lr=%.0E_es-lr=%.0E_batch-size=%d%s' % (
        args['--extractor-net-architecture'], args['--estimator-net-architecture'],
        float(args['--extractor-lr']), float(args['--estimator-lr']), int(args['--batch-size']), additional_info)
    hr_directory = os.path.join('/datagrid', 'personal', 'spetlrad', 'hr')

    logger.info(' %s' % basename)

    model_name, extractor_model, estimator_model, rgb, losses, epoch_shift = load_model_prepare_losses_file(
        args['--extractor-model-path'], args['--estimator-model-path']
    )
    train_loader, validation_loader, train_ds, val_ds = prepare_loaders(rgb)

    losses = evaluate_model(extractor_model, estimator_model, model_name, losses, epoch_shift)

    logger.info('Closing summary...')

    logger.info('Succesfully finished...')
