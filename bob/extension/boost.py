#!/usr/bin/env python
# encoding: utf-8
# Andre Anjos <andre.anjos@idiap.ch>
# Thu Mar 20 12:38:14 CET 2014

"""Helps looking for Boost on stock file-system locations"""

import os
import re
import sys
import glob
from distutils.version import LooseVersion

from .utils import uniq, egrep, find_header, find_library

def boost_version(version_hpp):

  matches = egrep(version_hpp, r"^#\s*define\s+BOOST_VERSION\s+(\d+)\s*$")
  if not len(matches): return None

  # we have a match, produce a string version of the version number
  version_int = int(matches[0].group(1))
  version_tuple = (
      version_int // 100000,
      (version_int // 100) % 1000,
      version_int % 100,
      )
  return '.'.join([str(k) for k in version_tuple])

class boost:
  """A class for capturing configuration information from boost

  Example usage:

  .. doctest::
     :options: +NORMALIZE_WHITESPACE +ELLIPSIS

     >>> from bob.extension import boost
     >>> pkg = boost('>= 1.35')
     >>> pkg.include_directory
     '...'
     >>> pkg.version
     '...'

  You can also use this class to retrieve information about installed Boost
  libraries and link information:

  .. doctest::
     :options: +NORMALIZE_WHITESPACE +ELLIPSIS

     >>> from bob.extension import boost
     >>> pkg = boost('>= 1.35')
     >>> pkg.libconfig(['python', 'system'])
     (...)

  """

  def __init__ (self, requirement=''):
    """
    Searches for the Boost library in stock locations. Allows user to override.

    If the user sets the environment variable BOB_PREFIX_PATH, that prefixes
    the standard path locations.
    """

    candidates = find_header('version.hpp', subpaths=['boost', 'boost?*'])

    if not candidates:
      raise RuntimeError("could not find boost's `version.hpp' - have you installed Boost on this machine?")

    found = False

    if not requirement:
      # since we use boost headers **including the boost/ directory**, we need to go one level lower
      self.include_directory = os.path.dirname(os.path.dirname(candidates[0]))
      self.version = boost_version(candidates[0])
      found = True

    else:

      # requirement is 'operator' 'version'
      operator, required = [k.strip() for k in requirement.split(' ', 1)]

      # now check for user requirements
      for path in candidates:
        version = boost_version(path)
        available = LooseVersion(version)
        if (operator == '<' and available < required) or \
           (operator == '<=' and available <= required) or \
           (operator == '>' and available > required) or \
           (operator == '>=' and available >= required) or \
           (operator == '==' and available == required):
          self.include_directory = path
          self.version = version
          found = True
          break

    if not found:
      raise RuntimeError("could not find the required (%s) version of boost on the file system (looked at: %s)" % (requirement, ', '.join(candidates)))

    # normalize
    self.include_directory = os.path.normpath(self.include_directory)


  def libconfig(self, modules, only_static=False,
      templates=['boost_%(name)s-mt-%(py)s', 'boost_%(name)s-%(py)s', 'boost_%(name)s-mt', 'boost_%(name)s']):
    """Returns a tuple containing the library configuration for requested
    modules.

    This function respects the path location where the include files for Boost
    are installed.

    Parameters:

    modules (list of strings)
      A list of string specifying the requested libraries to search for. For
      example, to search for `libboost_mpi.so`, pass only ``mpi``.

    static (bool)
      A boolean, indicating if we should try only to search for static versions
      of the libraries. If not set, any would do.

    templates (list of template strings)
      A list that defines in which order to search for libraries on the default
      search path, defined by ``self.include_directory``. Tune this list if you
      have compiled specific versions of Boost with support to multi-threading
      (``-mt``), debug (``-g``), STLPORT (``-p``) or required to insert
      compiler, the underlying thread API used or your own namespace.

      Here are the keywords you can use:

      %(name)s
        resolves to the module name you are searching for

      %(ver)s
        resolves to the current boost version string (e.g. ``'1.50.0'``)

      %(py)s
        resolves to the string ``'pyXY'`` where ``XY`` represent the major and
        minor versions of the current python interpreter.

      Example templates:

      * ``'boost_%(name)s-mt'``
      * ``'boost_%(name)s'``
      * ``'boost_%(name)s-gcc43-%(ver)s'``

    Returns:

    directories (list of strings)
      A list of directories indicating where the libraries are installed

    libs (list of strings)
      A list of strings indicating the names of the libraries you can use
    """

    # make the include header prefix preferential
    prefix = os.path.dirname(self.include_directory)

    py = 'py%d%d' % sys.version_info[:2]

    filenames = []
    for module in modules:
      candidates = []
      modnames = [k % dict(name=module, ver=self.version, py=py) for k in
          templates]

      for modname in modnames:
        candidates += find_library(modname, version=self.version,
            prefixes=[prefix], only_static=only_static)

      if not candidates:
        raise RuntimeError("cannot find required boost module `%s' - make sure boost is installed on `%s' and that this module is named %s on the filesystem" % (module, prefix, ' or '.join(modnames)))

      # take the first choice that includes the prefix (or the absolute first choice otherwise)
      index = 0
      for i, candidate in enumerate(candidates):
        if candidate.find(prefix) == 0:
          index = i
          break
      filenames.append(candidates[index])

    # libraries
    libraries = []
    for f in filenames:
      name, ext = os.path.splitext(os.path.basename(f))
      if ext in ['.so', '.a', '.dylib', '.dll']:
        libraries.append(name[3:]) #strip 'lib' from the name
      else: #link against the whole thing
        libraries.append(':' + os.path.basename(f))

    # library paths
    libpaths = [os.path.dirname(k) for k in filenames]

    return uniq(libpaths), uniq(libraries)

  def macros(self):
    """Returns package availability and version number macros

    This method returns a python list with 2 macros indicating package
    availability and a version number, using standard GNU compatible names.
    Example:

    .. doctest::
       :options: +NORMALIZE_WHITESPACE +ELLIPSIS

       >>> from bob.extension import boost
       >>> pkg = boost('>= 1.34')
       >>> pkg.macros()
       [('HAVE_BOOST', '1'), ('BOOST_VERSION', '"..."')]

    """
    return [('HAVE_BOOST', '1'), ('BOOST_VERSION', '"%s"' % self.version)]
