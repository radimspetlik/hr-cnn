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

'''Generate resutls from heart rate computation (%(version)s)
  
Usage:
  %(prog)s (hci | cohface | pure | ecg-fitness) [--protocol=<string>] [--subset=<string> ...]
           [--verbose ...] [--plot] [--outdir=<path>] [--indir=<path>] 
           [--overwrite]
           [--dbdir=<path>]

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
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import logging

__logging_format__ = '[%(levelname)s] %(message)s'
logging.basicConfig(format=__logging_format__)
logger = logging.getLogger("perf_log")

from docopt import docopt

version = pkg_resources.require('bob.rppg.base')[0].version

import numpy
import bob.io.base


def statistics(args, outdir, sq_err, errors, mean_error_percentage, inferred, ground_truth, additional_text=''):
    # compute global statistics

    mean_sq_err = sq_err / len(errors)
    rmse = float(numpy.sqrt(mean_sq_err).squeeze())
    rmse_text = "Root Mean Squared Error{0:s} = {1:.2f}".format(additional_text, rmse)
    mean_error_percentage /= len(errors)
    mean_error_percentage =  float(mean_error_percentage.squeeze())
    mean_err_percent_text = "Mean of error-rate percentage{0:s} = {1:.2f}".format(additional_text, mean_error_percentage)
    mean_absolute_error = float(numpy.abs(numpy.array(errors)).mean().squeeze())
    mean_absolute_error_text = "Mean absolute error{0:s} = {1:.2f}".format(additional_text, mean_absolute_error)
    from scipy.stats import pearsonr
    correlation, p = pearsonr(inferred, ground_truth)
    pearson_text = "Pearson's correlation{0:s} = {1:.2f}".format(additional_text, correlation)
    pearson_significance_text = "Pearson's correlation{0:s} significance = {1:.2f}".format(additional_text, p)

    subset = '' if len(args['--subset']) == 0 else args['--subset'][0]

    percentage_survived_text = "Percentage of files survived = {0:.2f}".format(1.0)

    logger.info("==================")
    logger.info("=== STATISTICS%s %s ===" % (additional_text, subset))
    logger.info(rmse_text)
    logger.info(mean_err_percent_text)
    logger.info(mean_absolute_error_text)
    logger.info(pearson_text)
    logger.info(pearson_significance_text)

    # statistics in a text file
    stats_filename = os.path.join(outdir, subset + additional_text + '-stats.txt')
    stats_file = open(stats_filename, 'w')
    stats_file.write(rmse_text + "\n")
    stats_file.write(mean_err_percent_text + "\n")
    stats_file.write(mean_absolute_error_text + "\n")
    stats_file.write(pearson_text + "\n")
    stats_file.write(pearson_significance_text + "\n")
    stats_file.write(percentage_survived_text + "\n")
    stats_file.close()

    if bool(args['--plot']):
        # scatter plot
        f = pyplot.figure()
        ax = f.add_subplot(1, 1, 1)
        ax.scatter(ground_truth, inferred)
        ax.plot([40, 110], [40, 110], 'r--', lw=2)
        pyplot.xlabel('Ground truth [bpm]')
        pyplot.ylabel('Estimated heart-rate [bpm]')
        ax.set_title('Scatter plot')
        scatter_file = os.path.join(outdir, subset + additional_text + '-scatter.pdf')
        f.savefig(scatter_file, bbox_inches='tight')
        pyplot.close('all')

        # histogram of error
        f2 = pyplot.figure()
        ax2 = f2.add_subplot(1, 1, 1)
        ax2.hist(errors, bins=50, )
        ax2.set_title('Distribution of the error')
        distribution_file = os.path.join(outdir, subset + additional_text + '-error_histogram.pdf')
        f2.savefig(distribution_file, bbox_inches='tight')
        pyplot.close('all')

        # distribution of HR
        f3 = pyplot.figure()
        ax3 = f2.add_subplot(1, 1, 1)
        histoargs = {'bins': 50, 'alpha': 0.5, 'histtype': 'bar', 'range': (30, 120)}
        pyplot.hist(ground_truth, label='Real HR', color='g', **histoargs)
        pyplot.hist(inferred, label='Estimated HR', color='b', **histoargs)
        pyplot.ylabel("Test set")
        distribution_file = os.path.join(outdir, subset + additional_text + '-HR_distribution.pdf')
        f3.savefig(distribution_file, bbox_inches='tight')
        pyplot.close('all')


def main(qf, metrics, user_input=None):
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
    if args['--verbose'] == 1:
        logging.getLogger("perf_log").setLevel(logging.INFO)
    elif args['--verbose'] >= 2:
        logging.getLogger("perf_log").setLevel(logging.DEBUG)

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
    outdir = args['--outdir']

    # if output dir exists and not overwriting, stop
    if os.path.exists(outdir) and not args['--overwrite']:
        logger.info("Skipping output `%s': already exists, use --overwrite to force an overwrite", outdir)
        sys.exit()
    else:
        bob.io.base.create_directories_safe(outdir)

    for metric in metrics:
        errors = []
        sq_err = 0
        mean_error_percentage = 0
        inferred = []
        ground_truth = []

        for obj in objects:
            # load the heart rate
            logger.debug("Loading computed heart rate from `%s'...", obj.path)

            hr_file = obj.make_path(args['--indir'], '-%s.hdf5' % metric)
            if qf is not None:
                hr_file = obj.make_path(args['--indir'], '-%s-qf=%02d.hdf5' % (metric, qf))
            try:
                hr = bob.io.base.load(hr_file)
            except (IOError, RuntimeError) as e:
                logger.warning("Skipping file `%s' (no heart rate file available)", obj.path)
                continue

            gt = [obj.load_heart_rate_in_bpm()]
            if hr.size > 1:
                logger.debug('Replacing simple GT by complicated one...')
                if hr[0].size == 1:
                    gt = numpy.array([hr[1]])
                    hr = numpy.array([hr[0]])
                else:
                    gt = hr[1]
                    hr = hr[0]

            for hr_idx, hr in enumerate(hr):
                hr = float(hr)
                logger.debug("Computed heart rate : {0}".format(hr))

                # load ground truth
                logger.debug("Real heart rate : {0}".format(float(gt[hr_idx])))
                ground_truth.append(float(gt[hr_idx]))
                inferred.append(hr)
                error = hr - float(gt[hr_idx])
                logger.debug("Error = {0}".format(error))
                errors.append(error)
                sq_err += error ** 2
                mean_error_percentage += numpy.abs(error) / float(gt[hr_idx])

        statistics(args, outdir, sq_err, errors, mean_error_percentage, inferred, ground_truth, additional_text='-' + metric)

    return 0
