#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


"""Interface definition for Bob's database driver of this database.

Building a driver goes through 2 steps:

1. Define an command line interface inheriting
   :py:class:`bob.db.base.driver.Interface`
2. Create an entry point on the package's ``setup.py`` with type ``bob.db``,
   containing a pointer to this interface

Once the two steps are in place, then the command-line utility will show your
database and allow you to interact with it via the command line.

"""

import os
import sys

from bob.db.base.driver import Interface as AbstractInterface


def _dumplist(args):
  """Dumps lists of files based on your criteria."""

  from .__init__ import Database
  db = Database()

  r = db.objects(group=args.group)

  output = sys.stdout
  if args.selftest:
    from bob.db.base.utils import null
    output = null()

  for f in r:
    output.write('%s\n' % f.make_path(
        directory=args.directory, extension=args.extension))

  return 0


def _checkfiles(args):
  """Checks the existence of the files based on your criteria."""

  from . import Database
  db = Database()

  r = db.objects(group=args.group)

  # go through all files, check if they are available
  bad = [f for f in r if not os.path.exists(f.make_path(
      directory=args.directory, extension=args.extension))]

  # report
  output = sys.stdout
  if args.selftest:
    from bob.db.base.utils import null
    output = null()

  if bad:
    for f in bad:
      output.write('Cannot find file "%s"\n' % f.make_path(
          directory=args.directory, extension=args.extension))
    output.write('%d files (out of %d) were not found at "%s"\n' %
                 (len(bad), len(r), args.directory))

  return 0


class Interface(AbstractInterface):
  """Bob Manager interface for the Samples Database"""

  def name(self):
    '''Returns a simple name for this database, w/o funny characters, spaces'''
    return 'samples'

  def files(self):
    '''Returns a python iterable with all auxiliary files needed.

    The values should be take w.r.t. where the python file that declares the
    database is sitting at. Use this method to return names of files that are
    not kept with the database and can be stored on a remote server.
    '''
    return []

  def version(self):
    '''Returns the current version number from Bob's build'''

    import pkg_resources  # part of setuptools
    version = pkg_resources.require("bob.db.base")[0].version
    return version

  def type(self):
    '''Returns the type of auxiliary files you have for this database

    If you return 'sqlite', then we append special actions such as 'dbshell'
    on 'bob_dbmanage.py' automatically for you. Otherwise, we don't.

    If you use auxiliary text files, just return 'text'. We may provide
    special services for those types in the future.

    Use the special name 'builtin' if this database is an integral part of Bob.
    '''

    return 'builtin'

  def add_commands(self, parser):
    """A few commands this database can respond to."""

    from argparse import SUPPRESS
    from . import __doc__ as docs

    subparsers = self.setup_parser(parser, "Samples dataset", docs)

    # add the dumplist command
    dump_parser = subparsers.add_parser(
        'dumplist', help="Dumps list of files based on your criteria")
    dump_parser.add_argument(
        '-d', '--directory', help="if given, this path will be prepended to "
        "every entry returned [default: <internal>]")
    dump_parser.add_argument('-e', '--extension', default='.png',
                             help="if given, this extension will be appended "
                             "to every entry returned.")
    dump_parser.add_argument(
        '-g', '--group', help="if given, this value will limit the output "
        "files to those belonging to a particular group.",
        choices=('train', 'test'))
    dump_parser.add_argument(
        '--self-test', dest="selftest", action='store_true', help=SUPPRESS)
    dump_parser.set_defaults(func=_dumplist)  # action

    # add the checkfiles command
    check_parser = subparsers.add_parser(
        'checkfiles', help="Check if the files exist, based on your criteria")
    check_parser.add_argument(
        '-d', '--directory', help="the path to the root directory to use "
        "[default: <internal>]")
    check_parser.add_argument('-e', '--extension', default=".png",
                              help="the extension appended to every sample "
                              "[default: %(default)s]")
    check_parser.add_argument(
        '-g', '--group', help="if given, this value will limit the output "
        "files to those belonging to a particular group.",
        choices=('train', 'test'))
    check_parser.add_argument(
        '--self-test', dest="selftest", action='store_true', help=SUPPRESS)
    check_parser.set_defaults(func=_checkfiles)  # action
