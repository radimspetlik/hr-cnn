#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

'''Implements a global configuration system for bob using json.'''

from collections import defaultdict
import json
import logging
import os

logger = logging.getLogger(__name__)

ENVNAME = 'BOBRC'
"""Name of environment variable to look for an alternative for the RC file"""

RCFILENAME = '~' + os.sep + '.bobrc'
"""Default name to be used for the RC file to load"""


def _get_rc_path():
  """Returns the path to the bob rc file.
  This method will return the path to **exactly** one (global) resource
  configuration file in this fixed order of preference:

  1. A path pointed by the environment variable BOBRC
  2. A file named :py:attr:`RCFILENAME` on your HOME directory

  Returns
  -------
  str
      The path to the rc file.
  """
  if 'BOBRC' in os.environ:
    path = os.environ['BOBRC']
  else:
    path = os.path.expanduser(RCFILENAME)

  return path


def _loadrc():
  '''Loads the default configuration file, or an override if provided

  This method will load **exactly** one (global) resource configuration file as
  returned by :py:func:`_get_rc_path`.

  Returns:

    dict: A dictionary of key-values representing the resolved context, after
    loading the provided modules and resolving all variables.

  '''

  def _default_none_dict(dct):
    dct2 = defaultdict(lambda: None)
    dct2.update(dct)
    return dct2

  path = _get_rc_path()
  if not os.path.exists(path):
    logger.debug("No RC file found")
    return _default_none_dict({})

  logger.debug("Loading RC file `%s'...", path)

  with open(path, 'rt') as f:
    return json.load(f, object_hook=_default_none_dict)


def _rc_to_str(context):
  """Converts the configurations into a pretty JSON formatted string.

  Parameters
  ----------
  context : dict
      All the configurations to save into the rc file.

  Returns
  -------
  str
      The configurations in a JSON formatted string.
  """

  return json.dumps(context, sort_keys=True, indent=4, separators=(',', ': '))


def _saverc(context):
  """Saves the context into the global rc file.

  Parameters
  ----------
  context : dict
      All the configurations to save into the rc file.
  """

  path = _get_rc_path()
  with open(path, 'wt') as f:
    f.write(_rc_to_str(context))
