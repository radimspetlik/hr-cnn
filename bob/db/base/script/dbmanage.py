#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue Jun 28 17:12:28 2011 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""This script drives all commands from the specific database subdrivers.
"""

epilog = """  For a list of available databases:
  >>> %(prog)s --help

  For a list of actions on a database:
  >>> %(prog)s <database-name> --help
"""

from ..manage import *

def main(user_input=None):

  from argparse import RawDescriptionHelpFormatter
  parser = create_parser(description=__doc__, epilog=epilog,
      formatter_class=RawDescriptionHelpFormatter)
  args = parser.parse_args(args=user_input)
  if hasattr(args, 'func'):
    return args.func(args)
  else:
    return parser.parse_args(args=['--help'])
