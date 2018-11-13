#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Sat 19 Oct 12:51:01 2013

"""Sets-up logging, centrally for Bob (including C++ code).
"""

import logging
from bob.extension.log import setup, set_verbosity_level
from ._logging import reset

# get the default root logger of Bob
_logger = logging.getLogger('bob')

# this will setup divergence from C++ into python.logging correctly
reset(logging.getLogger('bob.c++'))

# this will make sure we don't fiddle with python callables after termination.
# See: http://stackoverflow.com/questions/18184209/holding-python-produced-
# value-in-a-c-static-boostshared-ptr
import atexit
atexit.register(reset)
del atexit


def add_command_line_option(parser, short_option='-v'):
  """Adds the verbosity command line option to the given parser.

  The verbosity can by set to 0 (error), 1 (warning), 2 (info) or 3 (debug) by
  including the according number of --verbose command line arguments (e.g.,
  ``-vv`` for info level).

  Parameters
  ----------
  parser : :py:class:`argparse.ArgumentParser` or one of its derivatives
      A command line parser that you want to add a verbose option to
  short_option : :obj:`str`, optional
      The short command line option that should be used for increasing the
      verbosity. By default, ``'-v'`` is considered as the shortcut
  """
  parser.add_argument(
      short_option, '--verbose', action='count', default=0,
      help="Increase the verbosity level from 0 (only error messages) to 1 "
      "(warnings), 2 (log messages), 3 (debug information) by adding the "
      "--verbose option as often as desired (e.g. '-vvv' for debug).")


__all__ = [_ for _ in dir() if not _.startswith('_')]
