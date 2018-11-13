#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Commands
"""

import os
import sys
import pkg_resources

import numpy as np

import bob.io.base
import bob.db.base
from bob.db.base.driver import Interface as BaseInterface

import utils

from datasetworkers import *
from QRSDetectorOffline import QRSDetectorOffline


DATABASE_LOCATION = ''


# Driver API
# ==========

def dumplist(args):
    """Dumps lists of files on the database, inputs your arguments"""

    from . import Database
    db = Database()

    objects = db.objects()

    output = sys.stdout
    if args.selftest:
        from bob.db.base.utils import null
        output = null()

    for obj in objects:
        output.write('%s\n' % (obj.make_path(directory=args.directory, extension=args.extension),))

    return 0


def estimate_hr_in_bpm(hr_directory, db_name, basename):
    basename = basename.replace('.hdf5', '')
    ecg_data_raw, ecg_idx = ECGFitnessDatasetWorker.load_ecg(os.path.join(hr_directory, 'db', db_name, basename), 'c920')
    ecg_idx = ecg_idx[:, 1].astype('int32')

    qrs_log_dir = os.path.join(hr_directory, 'experiments', 'cnn', db_name, 'qrs-detector-bob', '/'.join(basename.split('/')[0:-1]))
    qrs_detections = QRSDetectorOffline(ecg_data_path="", ecg_data_raw=ecg_data_raw[ecg_idx[0]:ecg_idx[-1], 0:2], verbose=False,
                                        log_data=False, plot_data=True, show_plot=False, log_dir=qrs_log_dir).ecg_data_detected

    hr = qrs_detections[:, 2].sum()

    return hr


def create_bboxes():
    from __init__ import Database
    db = Database()

    objects = db.objects()
    basedir = pkg_resources.resource_filename(__name__, 'data')
    for obj in objects:
        output = obj.make_path(basedir + '/bbox/', '.face')
        outdir = os.path.dirname(output)
        if not os.path.exists(outdir): os.makedirs(outdir)
        # if os.path.exists(output):
        #     print("Skipping `%s' (meta file exists)" % obj.make_path())
        #     continue
        # try:
        #     print("Creating meta data for `%s'..." % obj.make_path())
        #     bb = obj.run_face_detector('/datagrid/personal/spetlrad/hr/db/pure/')
        #     with open(output, 'w') as file:
        #         file.write(json.dumps(bb))
        # except IOError as e:
        #     print("Skipping `%s': %s" % (obj.stem, str(e)))
        #     continue


def create_meta(args):
    """Runs the face detection, heart-rate estimation, save outputs at package"""

    # if not args.force:
    #     raise RuntimeError("This method will re-write the internal HDF5 files, " \
    #                        "which contain vital metadata used for generating results." \
    #                        " Make sure this is what you want to do reading the API for this " \
    #                        "package first (special attention to the method " \
    #                        ":py:meth:`File.run_face_detector`).")

    from __init__ import Database
    hr_directory = os.path.join('/datagrid', 'personal', 'spetlrad', 'hr')
    db_name = 'ecg-fitness'
    db_location = os.path.join(hr_directory, 'db', db_name)

    db = Database(db_location)

    objects = db.objects()
    if args.selftest:
        objects = objects[:5]
    if args.limit:
        objects = objects[:args.limit]

    if args.grid_count:
        print(len(objects))
        sys.exit(0)

    # if we are on a grid environment, just find what I have to process.
    if 'SGE_TASK_ID' in os.environ:
        pos = int(os.environ['SGE_TASK_ID']) - 1
        if pos >= len(objects):
            raise RuntimeError("Grid request for job %d on a setup with %d jobs" % \
                               (pos, len(objects)))
        objects = [objects[pos]]

    if args.selftest:
        basedir = pkg_resources.resource_filename(__name__, 'test-data')
    else:
        basedir = pkg_resources.resource_filename(__name__, 'data')

    for obj in objects:
        output = obj.make_path(basedir, '.hdf5')
        if os.path.exists(output) and not args.force:
            print("Skipping `%s' (meta file exists)" % obj.make_path())
            continue
        try:
            print("Creating meta data for `%s'..." % obj.make_path())
            kpts = obj.load_drmf_keypoints()
            bb = obj.run_face_detector(db_location, start_frame=1, max_frames=1)[0]
            hr = estimate_hr_in_bpm(hr_directory, db_name, obj.make_path())
            if bb and hr:
                outdir = os.path.dirname(output)
                if not os.path.exists(outdir): os.makedirs(outdir)
                if os.path.exists(output): os.remove(output)
                h5 = bob.io.base.HDF5File(output, 'a')
                h5.create_group('face_detector')
                h5.cd('face_detector')
                h5.set('topleft_x', bb.topleft[1])
                h5.set('topleft_y', bb.topleft[0])
                h5.set('width', bb.size[1])
                h5.set('height', bb.size[0])
                h5.cd('..')
                h5.set('heartrate', hr)
                h5.set('drmf_landmarks66', kpts)
                h5.set_attribute('units', 'beats-per-minute', 'heartrate')
                h5.close()
            else:
                print("Skipping `%s': Missing Heart-rate" % (obj.make_path(),))
                print(" -> Heart-rate  : %s" % hr)
                print(" -> Bounding box: %s" % bb)

        except IOError as e:
            print("Skipping `%s': %s" % (obj.make_path(), str(e)))
            continue

        finally:
            if args.selftest:
                if os.path.exists(basedir):
                    import shutil
                    shutil.rmtree(basedir)

    return 0


def debug(args):
    """Debugs the face detection and heart-rate estimation"""

    from . import Database
    db = Database()

    objects = db.objects()
    if args.selftest:
        objects = objects[:5]
    if args.limit:
        objects = objects[:args.limit]

    if args.grid_count:
        print(len(objects))
        sys.exit(0)

    # if we are on a grid environment, just find what I have to process.
    if os.environ.has_key('SGE_TASK_ID'):
        pos = int(os.environ['SGE_TASK_ID']) - 1
        if pos >= len(objects):
            raise RuntimeError("Grid request for job %d on a setup with %d jobs" % \
                               (pos, len(objects)))
        objects = [objects[pos]]

    basedir = 'debug'

    for obj in objects:
        print("Creating debug data for `%s'..." % obj.make_path())
        try:

            detections = obj.run_face_detector(args.directory)
            # save annotated video file
            output = obj.make_path(args.output_directory, '.avi')
            print("Annotating video `%s'" % output)
            utils.annotate_video(obj.load_video(args.directory), detections, output)

            print("Annotating heart-rate `%s'" % output)
            output = obj.make_path(args.output_directory, '.pdf')
            utils.explain_heartrate(obj, args.directory, output)

        except IOError as e:
            print("Skipping `%s': %s" % (obj.stem, str(e)))
            continue

        finally:
            if args.selftest:
                if os.path.exists(args.output_directory):
                    import shutil
                    shutil.rmtree(args.output_directory)

    return 0


def checkfiles(args):
    """Checks the existence of the files based on your criteria"""

    from . import Database
    db = Database()

    objects = db.objects()

    # go through all files, check if they are available on the filesystem
    good = []
    bad = []
    for obj in objects:
        if os.path.exists(obj.make_path(directory=args.directory, extension=args.extension)):
            good.append(obj)
        else:
            bad.append(obj)

    # report
    output = sys.stdout
    if args.selftest:
        from bob.db.base.utils import null
        output = null()

    if bad:
        for obj in bad:
            output.write('Cannot find file "%s"\n' % (obj.make_path(directory=args.directory, extension=args.extension),))
        output.write('%d files (out of %d) were not found at "%s"\n' % \
                     (len(bad), len(objects), args.directory))

    return 0


def write_kpts_and_hr(args):
    """Refactor metadata with DMRF and HR only"""
    from . import Database
    db = Database()

    basedir = pkg_resources.resource_filename(__name__, 'new-data')

    objects = db.objects()

    for obj in objects:
        kpts = obj.load_drmf_keypoints()
        hr = obj.load_heart_rate_in_bpm()

        output = obj.make_path(basedir, '.hdf5')
        if os.path.exists(output) and not args.force:
            print("Skipping `%s' (meta file exists)" % obj.make_path())
            continue
        try:
            print("Refactoring meta data for `%s'..." % obj.make_path())
            if kpts[0, 0] and hr:
                outdir = os.path.dirname(output)
                if not os.path.exists(outdir): os.makedirs(outdir)
                h5 = bob.io.base.HDF5File(output, 'w')
                h5.set('heartrate', hr)
                h5.set_attribute('units', 'beats-per-minute', 'heartrate')
                h5.close()
            else:
                print("Skipping `%s': Missing Heart-rate" % (obj.stem,))
                print(" -> Heart-rate  : %s" % hr)
        except IOError as e:
            print("Skipping `%s': %s" % (obj.stem, str(e)))
            continue

    return 0


class Interface(BaseInterface):

    def name(self):
        return 'pure'

    def files(self):
        basedir = pkg_resources.resource_filename(__name__, '')
        filelist = os.path.join(basedir, 'files.txt')
        return [os.path.join(basedir, k.strip()) for k in \
                open(filelist, 'rt').readlines() if k.strip()]

    def version(self):
        import pkg_resources
        return pkg_resources.require('bob.db.%s' % self.name())[0].version

    def type(self):
        return 'text'

    def add_commands(self, parser):
        """Add specific subcommands that the action "dumplist" can use"""

        from . import __doc__ as docs

        subparsers = self.setup_parser(parser, "PURE dataset", docs)

        from argparse import SUPPRESS

        # add the dumplist command
        dump_message = "Dumps list of files based on your criteria"
        dump_parser = subparsers.add_parser('dumplist', help=dump_message)
        dump_parser.add_argument('-d', '--directory', dest="directory", default=DATABASE_LOCATION,
                                 help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
        dump_parser.add_argument('-e', '--extension', dest="extension", default='',
                                 help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
        dump_parser.add_argument('--self-test', dest="selftest", default=False, action='store_true', help=SUPPRESS)
        dump_parser.set_defaults(func=dumplist)  # action

        # add the checkfiles command
        check_message = "Check if the files exist, based on your criteria"
        check_parser = subparsers.add_parser('checkfiles', help=check_message)
        check_parser.add_argument('-d', '--directory', dest="directory", default=DATABASE_LOCATION,
                                  help="if given, this path will be prepended to every entry returned (defaults to '%(default)s')")
        check_parser.add_argument('-e', '--extension', dest="extension", default='',
                                  help="if given, this extension will be appended to every entry returned (defaults to '%(default)s')")
        check_parser.add_argument('--self-test', dest="selftest", default=False, action='store_true', help=SUPPRESS)
        check_parser.set_defaults(func=checkfiles)  # action

        # add the create_meta command
        meta_message = create_meta.__doc__
        meta_parser = subparsers.add_parser('mkmeta', help=create_meta.__doc__)
        meta_parser.add_argument('-d', '--directory', dest="directory", default=DATABASE_LOCATION,
                                 help="This path points to the location where the database raw files are installed (defaults to '%(default)s')")
        meta_parser.add_argument('--grid-count', dest="grid_count", default=False, action='store_true', help=SUPPRESS)
        meta_parser.add_argument('--force', dest="force", default=False, action='store_true',
                                 help='If set, will overwrite existing meta files if they exist. Otherwise, just run on unexisting data')
        meta_parser.add_argument('--limit', dest="limit", default=0, type=int,
                                 help="Limits the number of objects to treat (defaults to '%(default)')")
        meta_parser.add_argument('--self-test', dest="selftest", default=False, action='store_true', help=SUPPRESS)
        meta_parser.set_defaults(func=create_meta)  # action

        # debug
        debug_message = debug.__doc__
        debug_parser = subparsers.add_parser('debug', help=debug.__doc__)
        debug_parser.add_argument('-d', '--directory', dest="directory", default=DATABASE_LOCATION,
                                  help="This path points to the location where the database raw files are installed (defaults to '%(default)s')")
        debug_parser.add_argument('-o', '--output-directory', dest="output_directory", default='debug',
                                  help="This path points to the location where the debugging results will be stored (defaults to '%(default)s')")
        debug_parser.add_argument('--grid-count', dest="grid_count", default=False, action='store_true', help=SUPPRESS)
        debug_parser.add_argument('--limit', dest="limit", default=0, type=int,
                                  help="Limits the number of objects to treat (defaults to '%(default)')")
        debug_parser.add_argument('--self-test', dest="selftest", default=False, action='store_true', help=SUPPRESS)
        debug_parser.set_defaults(func=debug)  # action

        # add the write_kpts_and_hr command
        meta_message = write_kpts_and_hr.__doc__
        meta_parser = subparsers.add_parser('write-kpts-hr', help=write_kpts_and_hr.__doc__)
        meta_parser.set_defaults(func=write_kpts_and_hr)  # action


def main_loop(job_id):
    class Antoine:
        pass

    args = Antoine()
    args.force = False
    args.selftest = False
    args.limit = False
    args.grid_count = False

    os.environ["SGE_TASK_ID"] = '%d' % job_id
    create_meta(args)


if __name__ == '__main__':
    from multiprocessing import Pool

    pool_size = 3
    p = Pool(pool_size)
    p.map(main_loop, range(1, 201))
