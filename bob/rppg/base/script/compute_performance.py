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

'''Generate resutls from heart rate computation (%(version)s)
  
Usage:
  %(prog)s (hci | cohface) [--protocol=<string>] [--subset=<string> ...] 
           [--verbose ...] [--plot] [--outdir=<path>] [--indir=<path>] 
           [--overwrite] 

  %(prog)s (--help | -h)
  %(prog)s (--version | -V)


Options:
  -h, --help                Show this help message and exit
  -V, --version             Show version
  -v, --verbose             Increases the verbosity (may appear multiple times)
  -P, --plot                Set this flag if you'd like to see some plots. 
  -p, --protocol=<string>   Protocol [default: all].
  -s, --subset=<string>     Data subset to load. If nothing is provided 
                            all the data sets will be loaded.
  -i, --indir=<path>        The path to the saved heart rate values on your disk. 
  -d, --outdir=<path>       The path to the output directory where the results
                            will be stored [default: performances].
  -O, --overwrite           By default, we don't overwrite existing files. 
                            Set this flag if you want to overwrite existing files.

Examples:

  To run the results generation for the cohface database

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
logger = logging.getLogger("perf_log")

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
      version='Results for videos (%s)' % version,
      )

  # if the user wants more verbosity, lowers the logging level
  if args['--verbose'] == 1: logging.getLogger("perf_log").setLevel(logging.INFO)
  elif args['--verbose'] >= 2: logging.getLogger("perf_log").setLevel(logging.DEBUG)

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

  # errors
  errors = []
  rmse = 0;
  mean_error_percentage = 0

  inferred = []
  ground_truth = []

  ################
  ### LET'S GO ###
  ################
  outdir = args['--outdir']
  
  # if output dir exists and not overwriting, stop 
  if os.path.exists(outdir) and not args['--overwrite']:
    logger.info("Skipping output `%s': already exists, use --overwrite to force an overwrite", outdir)
    sys.exit()
  else: 
    bob.io.base.create_directories_safe(outdir)

  for obj in objects:

    # load the heart rate 
    logger.debug("Loading computed heart rate from `%s'...", obj.path)
    hr_file = obj.make_path(args['--indir'], '.hdf5')
    try:
      hr = bob.io.base.load(hr_file)
    except (IOError, RuntimeError) as e:
      logger.warn("Skipping file `%s' (no heart rate file available)", obj.path)
      continue

    hr = hr[0]
    logger.debug("Computed heart rate : {0}".format(hr))

    # load ground truth
    gt = obj.load_heart_rate_in_bpm()
    logger.debug("Real heart rate : {0}".format(gt))
    ground_truth.append(gt)
    inferred.append(hr)
    error = hr - gt
    logger.debug("Error = {0}".format(error))
    errors.append(error)
    rmse += error**2
    mean_error_percentage += numpy.abs(error)/gt

  # compute global statistics 
  rmse /= len(errors)
  rmse = numpy.sqrt(rmse)
  rmse_text = "Root Mean Squared Error = {0:.2f}". format(rmse)
  mean_error_percentage /= len(errors)
  mean_err_percent_text = "Mean of error-rate percentage = {0:.2f}". format(mean_error_percentage)
  from scipy.stats import pearsonr
  correlation, p = pearsonr(inferred, ground_truth)
  pearson_text = "Pearson's correlation = {0:.2f} ({1:.2f} significance)". format(correlation, p)
 
  logger.info("==================")
  logger.info("=== STATISTICS ===")
  logger.info(rmse_text)
  logger.info(mean_err_percent_text)
  logger.info(pearson_text)

  # statistics in a text file
  stats_filename = os.path.join(outdir, 'stats.txt')
  stats_file = open(stats_filename, 'w')
  stats_file.write(rmse_text + "\n")
  stats_file.write(mean_err_percent_text + "\n")
  stats_file.write(pearson_text + "\n")
  stats_file.close()

  # scatter plot
  from matplotlib import pyplot
  f = pyplot.figure()
  ax = f.add_subplot(1,1,1)
  ax.scatter(ground_truth, inferred)
  ax.plot([40, 110], [40, 110], 'r--', lw=2)
  pyplot.xlabel('Ground truth [bpm]')
  pyplot.ylabel('Estimated heart-rate [bpm]')
  ax.set_title('Scatter plot')
  scatter_file = os.path.join(outdir, 'scatter.png')
  pyplot.savefig(scatter_file)

  # histogram of error
  f2 = pyplot.figure()
  ax2 = f2.add_subplot(1,1,1)
  ax2.hist(errors, bins=50, )
  ax2.set_title('Distribution of the error')
  distribution_file = os.path.join(outdir, 'error_distribution.png')
  pyplot.savefig(distribution_file)

  # distribution of HR
  f3 = pyplot.figure()
  ax3 = f2.add_subplot(1,1,1)
  histoargs = {'bins': 50, 'alpha': 0.5, 'histtype': 'bar', 'range': (30, 120)} 
  pyplot.hist(ground_truth, label='Real HR', color='g', **histoargs)
  pyplot.hist(inferred, label='Estimated HR', color='b', **histoargs)
  pyplot.ylabel("Test set")

  if bool(args['--plot']):
    pyplot.show()
  
  return 0
