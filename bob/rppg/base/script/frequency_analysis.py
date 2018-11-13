#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Guillaume Heusch <guillaume.heusch@idiap.ch>,
# 
# This file is part of bob.rpgg.base.
# 
# bob.rppg.base is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# bob.rppg.base is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with bob.rppg.base. If not, see <http://www.gnu.org/licenses/>.

'''Frequency analysis of the filtered color signals to get the heart-rate (%(version)s)

Usage:
  %(prog)s (cohface | hci) [--protocol=<string>] [--subset=<string> ...]  
           [--verbose ...] [--plot] [--indir=<path>] [--outdir=<path>] 
           [--framerate=<int>] [--nsegments=<int>] [--nfft=<int>] 
           [--overwrite] 

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
  -o, --outdir=<path>       The path to the output directory where the resulting
                            color signals will be stored [default: heart-rate].
  -O, --overwrite           By default, we don't overwrite existing files. The
                            processing will skip those so as to go faster. If you
                            still would like me to overwrite them, set this flag.
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
__logging_format__='[%(levelname)s] %(message)s'
logging.basicConfig(format=__logging_format__)
logger = logging.getLogger("hr_log")

from docopt import docopt

version = pkg_resources.require('bob.rppg.base')[0].version

import numpy
import bob.io.base

def main(user_input=None):

  # Parse the command-line arguments
  if user_input is not None:
      arguments = user_input
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
      version='Frequency analysis for videos (%s)' % version,
      )

  # if the user wants more verbosity, lowers the logging level
  if args['--verbose'] == 1: logging.getLogger("hr_log").setLevel(logging.INFO)
  elif args['--verbose'] >= 2: logging.getLogger("hr_log").setLevel(logging.DEBUG)

  # chooses the database driver to use
  if args['cohface']:
    import bob.db.cohface
    if os.path.isdir(bob.db.cohface.DATABASE_LOCATION):
      logger.debug("Using Idiap default location for the DB")
      dbdir = bob.db.cohface.DATABASE_LOCATION
    elif args['--indir'] is not None:
      logger.debug("Using provided location for the DB")
      dbdir = args['--indir']
    else:
      logger.warn("Could not find the database directory, please provide one")
      sys.exit()
    db = bob.db.cohface.Database(dbdir)
    if not((args['--protocol'] == 'all') or (args['--protocol'] == 'clean') or (args['--protocol'] == 'natural')):
      logger.warning("Protocol should be either 'clean', 'natural' or 'all' (and not {0})".format(args['--protocol']))
      sys.exit()
    objects = db.objects(args['--protocol'], args['--subset'])

  elif args['hci']:
    import bob.db.hci_tagging
    import bob.db.hci_tagging.driver
    if os.path.isdir(bob.db.hci_tagging.driver.DATABASE_LOCATION):
      logger.debug("Using Idiap default location for the DB")
      dbdir = bob.db.hci_tagging.driver.DATABASE_LOCATION
    elif args['--indir'] is not None:
      logger.debug("Using provided location for the DB")
      dbdir = args['--indir'] 
    else:
      logger.warn("Could not find the database directory, please provide one")
      sys.exit()
    db = bob.db.hci_tagging.Database()
    if not((args['--protocol'] == 'all') or (args['--protocol'] == 'cvpr14')):
      logger.warning("Protocol should be either 'all' or 'cvpr14' (and not {0})".format(args['--protocol']))
      sys.exit()
    objects = db.objects(args['--protocol'], args['--subset'])

  ################
  ### LET'S GO ###
  ################
  for obj in objects:

    # expected output file
    output = obj.make_path(args['--outdir'], '.hdf5')

    # if output exists and not overwriting, skip this file
    if os.path.exists(output) and not args['--overwrite']:
      logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
      continue

    # load the filtered color signals of shape (3, nb_frames)
    logger.info("Frequency analysis of color signals from `%s'...", obj.path)
    filtered_file = obj.make_path(args['--indir'], '.hdf5')
    try:
      signal = bob.io.base.load(filtered_file)
    except (IOError, RuntimeError) as e:
      logger.warn("Skipping file `%s' (no color signals file available)", obj.path)
      continue

    if bool(args['--plot']):
      from matplotlib import pyplot
      pyplot.plot(range(signal.shape[0]), signal, 'g')
      pyplot.title('Filtered green signal')
      pyplot.show()

    # find the segment length, such that we have 8 50% overlapping segments (Matlab's default)
    segment_length = (2*signal.shape[0]) // (int(args['--nsegments']) + 1) 

    # the number of points for FFT should be larger than the segment length ...
    if int(args['--nfft']) < segment_length:
      logger.warn("Skipping file `%s' (nfft < nperseg)", obj.path)
      continue

    nfft = int(args['--nfft'])
    from scipy.signal import welch
    green_f, green_psd = welch(signal, int(args['--framerate']), nperseg=segment_length, nfft=nfft)

    # find the max of the frequency spectrum in the range of interest
    first = numpy.where(green_f > 0.7)[0]
    last = numpy.where(green_f < 4)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)
    max_idx = numpy.argmax(green_psd[range_of_interest])
    f_max = green_f[range_of_interest[max_idx]]
    hr = f_max*60.0
    logger.info("Heart rate = {0}".format(hr))

    if bool(args['--plot']):
      from matplotlib import pyplot
      pyplot.semilogy(green_f, green_psd, 'g')
      xmax, xmin, ymax, ymin = pyplot.axis()
      pyplot.vlines(green_f[range_of_interest[max_idx]], ymin, ymax, color='red')
      pyplot.title('Power spectrum of the green signal (HR = {0:.1f})'.format(hr))
      pyplot.show()

    output_data = numpy.array([hr], dtype='float64')

    # saves the data into an HDF5 file with a '.hdf5' extension
    outdir = os.path.dirname(output)
    if not os.path.exists(outdir): bob.io.base.create_directories_safe(outdir)
    bob.io.base.save(output_data, output)
    logger.info("Output file saved to `%s'...", output)

  return 0
