#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Guillaume Heusch <guillaume.heusch@idiap.ch>,
# 
# This file is part of experiments.rpgg.base.
# 
# experiments.rppg.base is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# experiments.rppg.base is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with experiments.rppg.base. If not, see <http://www.gnu.org/licenses/>.

'''Frequency analysis of the filtered color signals to get the heart-rate (%(version)s)

Usage:
  %(prog)s (cohface | hci | pure | ecg-fitness) [--protocol=<string>] [--subset=<string> ...]
           [--verbose ...] [--plot] [--indir=<path>] [--outdir=<path>] 
           [--framerate=<int>] [--nsegments=<int>] [--nfft=<int>]
           [--prediction-interval-lambda=<float>]
           [--overwrite]
           [--stats-outdir=<path>]
           [--dbdir=<path>]

  %(prog)s (--help | -h)
  %(prog)s (--version | -V)


Options:
  -h, --help                Show this help message and exit
  -v, --verbose             Increases the verbosity (may appear multiple times)
  -V, --version             Show version
  -P, --plot                Set this flag if you'd like to follow-up the algorithm
                            execution graphically. We'll plot some interactions.
  -p, --protocol=<string>   Protocol [default: all].
  -s, --subset=<string>     Data subset to load. If nothing is provided 
                            all the data sets will be loaded.
  -i, --indir=<path>        The path to the saved filtered signals on your disk
                            [default: filtered].
  -l, --prediction-interval-lambda Prediction interval lambda [default: 0.0]
  -o, --outdir=<path>       The path to the output directory where the resulting
                            color signals will be stored [default: heart-rate].
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  --stats-outdir=<path>     Path to stats dir output...
  -f, --framerate=<int>     Frame-rate of the video sequence [default: 61]
  --nsegments=<int>         Number of overlapping segments in Welch procedure
                            [default: 12].
  --nfft=<int>              Number of points to compute the FFT [default: 8192].

Examples:

  To run the frequency analysis for the cohface database

    $ %(prog)s cohface -v

  You can change the output directory using the `-o' flag:

    $ %(prog)s hci -v -o /path/to/result/directory


See '%(prog)s --help' for more information.

'''

import os
import sys
import pkg_resources
import logging
from cmp.nrppg.db.datasetworkers import ECGFitnessDatasetWorker, Dataset
from clab.QRSDetectorOffline import QRSDetectorOffline

__logging_format__ = '[%(levelname)s] %(message)s'
logging.basicConfig(format=__logging_format__)
logger = logging.getLogger("hr_log")

from docopt import docopt

import numpy
import torch
from torch.autograd import Variable
from cmp.nrppg.torch.TorchLossComputer import TorchLossComputer


def main(estimator_model, job_id, pool_size, metrics, hr_directory, fps, user_input=None):
    # Parse the command-line arguments
    if user_input is not None:
        arguments = user_input
    else:
        arguments = sys.argv[1:]

    prog = os.path.basename(sys.argv[0])
    completions = dict(
        prog=prog,
        version='custom bob.rppg.base',
    )
    args = docopt(
        __doc__ % completions,
        argv=arguments,
        version='Frequency analysis for videos (custom bob.rppg.base)',
    )

    # if the user wants more verbosity, lowers the logging level
    if args['--verbose'] == 1:
        logging.getLogger("hr_log").setLevel(logging.INFO)
    elif args['--verbose'] >= 2:
        logging.getLogger("hr_log").setLevel(logging.DEBUG)

    # chooses the database driver to use

    db_name = ''
    if args['cohface']:
        db_name = 'cohface'
        import bob.db.cohface
        if os.path.isdir(bob.db.cohface.DATABASE_LOCATION):
            logger.debug("Using Idiap default location for the DB")
            dbdir = bob.db.cohface.DATABASE_LOCATION
        elif args['--dbdir'] is not None:
            logger.debug("Using provided location for the DB")
            dbdir = args['--dbdir']
        else:
            logger.warn("Could not find the database directory, please provide one")
            sys.exit()
        db = bob.db.cohface.Database(dbdir)
        if not ((args['--protocol'] == 'all') or (args['--protocol'] == 'clean') or (args['--protocol'] == 'natural')):
            logger.warning("Protocol should be either 'clean', 'natural' or 'all' (and not {0})".format(args['--protocol']))
            sys.exit()
        objects = db.objects(args['--protocol'], args['--subset'])

    elif args['hci']:
        db_name = 'hci'
        import bob.db.hci_tagging
        import bob.db.hci_tagging.driver
        if os.path.isdir(bob.db.hci_tagging.driver.DATABASE_LOCATION):
            logger.debug("Using Idiap default location for the DB")
            dbdir = bob.db.hci_tagging.driver.DATABASE_LOCATION
        elif args['--dbdir'] is not None:
            logger.debug("Using provided location for the DB")
            dbdir = args['--dbdir']
        else:
            logger.warn("Could not find the database directory, please provide one")
            sys.exit()
        db = bob.db.hci_tagging.Database()
        if not ((args['--protocol'] == 'all') or (args['--protocol'] == 'cvpr14')):
            logger.warning("Protocol should be either 'all' or 'cvpr14' (and not {0})".format(args['--protocol']))
            sys.exit()
        objects = db.objects(args['--protocol'], args['--subset'])

    elif args['pure']:
        db_name = 'pure'
        import bob.db.pure
        import bob.db.pure.driver
        if os.path.isdir(bob.db.pure.driver.DATABASE_LOCATION):
            logger.debug("Using Idiap default location for the DB")
            dbdir = bob.db.pure.driver.DATABASE_LOCATION
        elif args['--dbdir'] is not None:
            logger.debug("Using provided location for the DB")
            dbdir = args['--dbdir']
        else:
            logger.warning("Could not find the database directory, please provide one")
            sys.exit()
        db = bob.db.pure.Database(dbdir)
        if not ((args['--protocol'] == 'all')):
            logger.warning("Protocol should be 'all' (and not {0})".format(args['--protocol']))
            sys.exit()
        objects = db.objects(args['--protocol'], args['--subset'])
    elif args['ecg-fitness']:
        import bob.db.ecg_fitness
        import bob.db.ecg_fitness.driver

        if os.path.isdir(bob.db.ecg_fitness.driver.DATABASE_LOCATION):
            logger.debug("Using Idiap default location for the DB")
            dbdir = bob.db.ecg_fitness.driver.DATABASE_LOCATION
        elif args['--dbdir'] is not None:
            logger.debug("Using provided location for the DB")
            dbdir = args['--dbdir']
        else:
            logger.warn("Could not find the database directory, please provide one")
            sys.exit()
        db = bob.db.ecg_fitness.Database(dbdir)
        objects = db.objects(args['--protocol'], args['--subset'])

    ################
    ### LET'S GO ###
    ################
    computed_count = 0
    for obj_idx, obj in enumerate(objects):
        if obj_idx % pool_size != job_id:
            continue

        # expected output file
        output = {}
        for metric in metrics:
            output[metric] = obj.make_path(args['--outdir'], '-%s.hdf5' % metric)

        # if output exists and not overwriting, skip this file
        if os.path.exists(output['whole']) and not args['--overwrite']:
            logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output['whole'])
            continue

        computed_count += 1

        # load the filtered color signals of shape (3, nb_frames)
        logger.debug("Frequency analysis of color signals from `%s'...", obj.path)
        filtered_file = obj.make_path(args['--indir'], '.hdf5')
        try:
            signal = bob.io.base.load(filtered_file)
        except (IOError, RuntimeError) as e:
            logger.warning("Skipping file `%s' (no color signals file available)", obj.path)
            continue

        # find the max of the frequency spectrum in the range of interest
        signal_in_torch = Variable(torch.FloatTensor(signal)).cuda()

        gt_hr_bpm = obj.load_heart_rate_in_bpm()

        sq_errs = {}
        abs_errs = {}
        for metric in metrics:
            sq_errs[metric] = []
            abs_errs[metric] = []

        if 'am' in metrics:
            hr_am = float(TorchLossComputer.hr_bpm(signal_in_torch, float(args['--framerate']), cuda=True))
            sq_errs['am'].append((hr_am - gt_hr_bpm) ** 2)
            abs_errs['am'].append(abs(hr_am - gt_hr_bpm))

        signal_in_torch = signal_in_torch.view(1, 1, -1)
        hr = estimator_model(signal_in_torch).cpu().squeeze().data[0]
        sq_errs['whole'].append((hr - gt_hr_bpm) ** 2)
        abs_errs['whole'].append(abs(hr - gt_hr_bpm))

        window_length_s = 10
        window_length = int(fps * window_length_s)
        window_length = min(window_length, signal_in_torch.size()[2])

        hrs = None
        gts = None
        if args['ecg-fitness']:
            basename = obj.make_path().replace('.hdf5', '')
            ecg_data_raw, ecg_idx = ECGFitnessDatasetWorker.load_ecg(os.path.join(hr_directory, 'db', Dataset.ECG_FITNESS, basename), 'c920')
            ecg_idx = ecg_idx[:, 1].astype('int32')

        for shift in range(0, signal_in_torch.size()[2] - window_length, window_length):
            if args['ecg-fitness']:
                qrs_detections = QRSDetectorOffline(ecg_data_path="",
                                                    ecg_data_raw=ecg_data_raw[ecg_idx[shift]:ecg_idx[shift + window_length], 0:2],
                                                    verbose=False, log_data=False, plot_data=False, show_plot=False).ecg_data_detected

                local_hr_gt = qrs_detections[:, 2].sum() * (60.0 / window_length_s)
            else:
                local_hr_gt = gt_hr_bpm

            local_hr = estimator_model(signal_in_torch[:, :, shift:shift + window_length]).cpu().squeeze().data[0]
            if hrs is None:
                hrs = numpy.array(local_hr)
                gts = numpy.array(local_hr_gt)
            else:
                hrs = numpy.vstack((hrs, local_hr))
                gts = numpy.vstack((gts, local_hr_gt))

            if 'Parts' in metrics:
                sq_errs['Parts'].append((local_hr - local_hr_gt) ** 2)
                abs_errs['Parts'].append(abs(local_hr - local_hr_gt))

        if hrs is None:
            logger.warning('Skipping file %s (no HR signal available)')
            continue

        logger.info("Heart rate = {0}".format(hr))

        output_data = {}
        output_data['whole'] = numpy.array([hr], dtype='float64')

        # saves the data into an HDF5 file with a '.hdf5' extension
        for metric in metrics:
            outdir = os.path.dirname(output[metric])
            if not os.path.exists(outdir):
                bob.io.base.create_directories_safe(outdir)
            bob.io.base.save(output_data[metric], output[metric])
            logger.info("Output file saved to `%s'...", output[metric])

    if computed_count > 0 and len(sq_errs['whole']) > 0:
        stats_output = args['--stats-outdir'] + '%s.npy' % (db_name)
        outdir = os.path.dirname(stats_output)
        if not os.path.exists(outdir):
            bob.io.base.create_directories_safe(outdir)
        numpy.save(stats_output, {'sq_errs': sq_errs, 'abs_errs': abs_errs})

    return 0