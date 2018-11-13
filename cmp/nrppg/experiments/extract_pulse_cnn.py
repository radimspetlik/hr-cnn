#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Guillaume Heusch <guillaume.heusch@idiap.ch>,
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

"""Pulse extraction for database videos (%(version)s)

Usage:
  %(prog)s (cohface | hci | pure | ecg-fitness) [--cnn-model-path=<string>] [--protocol=<string>] [--subset=<string> ...]
           [--dbdir=<path>] [--outdir=<path>]
           [--batch-size=<int>] [--enable-cuda=<BOOL>] [--use-gpu=<BOOL>]
           [--faces-dir=<path>]
           [--start=<int>] [--end=<int>]
           [--framerate=<int>] [--order=<int>]
           [--window=<int>] [--gridcount]
           [--overwrite] [--verbose ...]
           [--plot]

  %(prog)s (--help | -h)
  %(prog)s (--version | -V)


Arguments:
    (cohface | hci | pure | ecg-fitness)         Database driver selection.

Options:
  -h, --help                Show this screen
  -V, --version             Show version
  -m, --cnn-model-path=<string>      Path to the CNN pytorch model.
  -p, --protocol=<string>   Protocol [default: all].
  -s, --subset=<string>     Data subset to load. If nothing is provided
                            all the data sets will be loaded.
  -d, --dbdir=<path>        The path to the database on your disk. If not set,
                            defaults to Idiap standard locations.
  -d, --faces-dir=<path>    The path to the extracted faces on your disk. If not set,
                            defaults to faces standard locations.
  -o, --outdir=<path>       The path to the directory where signal extracted
                            from the face area will be stored [default: pulse]
  -b, --batch-size=<int>    Size of the batch [default: 400]
  -c, --enable-cuda=<BOOL>  Is cuda enabled? [default: True]
  -g, --use-gpu=<BOOL>      Use GPU? [default: True]
  --start=<int>             Starting frame index [default: -1].
  --end=<int>               End frame index [default: -1].
  --framerate=<int>         Framerate of the video sequence [default: 61]
  --order=<int>             Order of the bandpass filter [default: 128]
  --window=<int>            Window size in the overlap-add procedure. A window
                            of zero means no procedure applied [default: 0].
  --gridcount               Tells the number of objects that will be processed.
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
  -v, --verbose             Increases the verbosity (may appear multiple times)
  -P, --plot                Set this flag if you'd like to follow-up the algorithm
                            execution graphically. We'll plot some interactions.


Example:

  To run the pulse extraction for the cohface database

    $ %(prog)s cohface --cnn-model-path -v

See '%(prog)s --help' for more information.

"""

import sys

from cmp.nrppg.cnn.ModelLoader import ModelLoader

import site  # initializes site properly

from docopt import docopt
import torch
import torch.utils.data as torch_data
from torch.autograd import Variable
from cmp.nrppg.cnn.dataset.FaceDatasetHdf5 import FaceDatasetHdf5
import numpy as np
import os.path
import pkg_resources
import logging

__logging_format__ = '[%(levelname)s] %(message)s'
logging.basicConfig(format=__logging_format__, stream=sys.stdout)
logger = logging.getLogger("extract_log")

version = pkg_resources.require('bob.rppg.base')[0].version


def main(extractor_model, rgb, job_id, jobs, user_input=None):
    # Parse the command-line arguments
    if user_input is not None:
        arguments = user_input
        # if not isinstance(arguments, list):
        #     arguments = arguments.split(' ')
    else:
        arguments = sys.argv[1:]

    prog = os.path.basename(sys.argv[0])
    completions = dict(
        prog=prog,
        version=version,
    )
    args = docopt(
        __doc__ % completions,
        argv=arguments,
        version='Signal extractor for videos (%s)' % version,
    )

    # if the user wants more verbosity, lowers the logging level
    if args['--verbose'] == 1:
        logging.getLogger("extract_log").setLevel(logging.INFO)
    elif args['--verbose'] >= 2:
        logging.getLogger("extract_log").setLevel(logging.DEBUG)

    # chooses the database driver to use
    if args['cohface']:
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
        import bob.db.pure
        import bob.db.pure.driver

        if os.path.isdir(bob.db.pure.driver.DATABASE_LOCATION):
            logger.debug("Using Idiap default location for the DB")
            dbdir = bob.db.pure.driver.DATABASE_LOCATION
        elif args['--dbdir'] is not None:
            logger.debug("Using provided location for the DB")
            dbdir = args['--dbdir']
        else:
            logger.warn("Could not find the database directory, please provide one")
            sys.exit()
        db = bob.db.pure.Database(dbdir)
        if not (args['--protocol'] == 'all'):
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

    # tells the number of grid objects, and exit
    if args['--gridcount']:
        print(len(objects))
        sys.exit()

    # if we are on a grid environment, just find what I have to process.
    if 'SGE_TASK_ID' in os.environ:
        pos = int(os.environ['SGE_TASK_ID']) - 1
        if pos >= len(objects):
            raise RuntimeError("Grid request for job %d on a setup with %d jobs" % (pos, len(objects)))
        objects = [objects[pos]]

    size = int(len(objects) / jobs)
    from_idx = size * job_id
    to_idx = size * (job_id + 1)
    objects = objects[from_idx:min(len(objects), to_idx)]

    # does the actual work - for every video in the available dataset,
    # extract the signals and dumps the results to the corresponding directory
    for key, obj in enumerate(objects):

        # expected output file
        output_path = obj.make_path(args['--outdir'], '.hdf5')

        # if output exists and not overwriting, skip this file
        if os.path.exists(output_path) and not args['--overwrite']:
            logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output_path)
            continue

        # load extracted faces
        faces_filepath = obj.make_path(args['--faces-dir'], '.h5')
        if not os.path.isfile(faces_filepath):
            logger.error("Missing file %s..." % faces_filepath)
            continue

        logger.info("Processing faces [%d/%d] from `%s'..." % (key+1, len(objects), faces_filepath))

        faceDB = FaceDatasetHdf5(faces_filepath, None, int(args['--batch-size']), False, True, rgb)
        train_loader = torch_data.DataLoader(faceDB, batch_size=int(args['--batch-size']), shuffle=False)

        outputs = None
        for batch_idx, (data, target) in enumerate(train_loader):
            # if (batch_idx + 1) * int(args['--batch-size']) < int(args['--start']):
            #     continue
            if 0 <= int(args['--end']) < batch_idx * int(args['--batch-size']):
                break

            data, target = Variable(data, volatile=True), Variable(target, volatile=True)
            if args['--use-gpu'] == 'True' and args['--enable-cuda'] == 'True':
                data, target = data.cuda(async=True), target.cuda(async=True)
            batch_output = extractor_model(data)
            # print("[Batch %d] done..." % (batch_idx))
            if outputs is None:
                Fs = faceDB.get_fps(batch_idx * int(args['--batch-size']))
                outputs = batch_output
            else:
                outputs = torch.cat((outputs, batch_output), dim=0)

        if outputs is None:
            logger.warning('File %s has no contents...' % faces_filepath)
            continue

        outputs = outputs.data.cpu().numpy().reshape(-1)

        N = outputs.shape[0]

        # indices where to start and to end the processing
        logger.debug("Sequence length = {0}".format(N))
        start_index = int(args['--start'])
        if start_index < 0:
            start_index = 0
        end_index = int(args['--end'])
        if (end_index < 0):
            end_index = N
        if end_index > N:
            logger.warn("Skipping Sequence {0} : not long enough ({1})".format(obj.path, N))
            continue

        # number of final frames
        nb_frames = N
        if end_index > 0:
            nb_frames = end_index - start_index

        # output data
        cnn_output = np.zeros(nb_frames, dtype='float64')

        # loop on video frames
        counter = 0
        for frame_id, signal in enumerate(outputs):

            if start_index <= frame_id < end_index:
                logger.debug("Processing output %d/%d...", frame_id + 1, end_index)

                cnn_output[counter] = signal

                counter += 1

            elif frame_id > end_index:
                break

        # build the final pulse signal
        pulse = cnn_output

        output_data = pulse

        # saves the data into an HDF5 file with a '.hdf5' extension
        outdir = os.path.dirname(output_path)
        if not os.path.exists(outdir): bob.io.base.create_directories_safe(outdir)
        bob.io.base.save(output_data, output_path)
        logger.info("Output file saved to `%s'...", output_path)