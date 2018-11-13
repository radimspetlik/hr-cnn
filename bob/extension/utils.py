#!/usr/bin/env python
# encoding: utf-8
# Andre Anjos <andre.dos.anjos@gmail.com>
# Fri 21 Mar 2014 10:37:40 CET

'''General utilities for building extensions'''

import os
import re
import sys
import glob
import platform
import pkg_resources
from . import DEFAULT_PREFIXES


def construct_search_paths(prefixes=None, subpaths=None, suffix=None):
  """Constructs a list of candidate paths to search for.

  The list of paths is constructed using the following order of priority:

  1. ``BOB_PREFIX_PATH`` environment variable, if set. ``BOB_PREFIX_PATH`` can
     contain several paths divided by :any:`os.pathsep`.
  2. The paths provided with the ``prefixes`` parameter.
  3. The current python executable prefix.
  4. The ``CONDA_PREFIX`` environment variable, if set.
  5. :any:`DEFAULT_PREFIXES`.

  Parameters
  ----------
  prefixes : [:obj:`str`], optional
      The list of paths to be added to the results.
  subpaths : [:obj:`str`], optional
      A list of subpaths to be appended to each path at the end. For
      example, if you specify ``['foo', 'bar']`` for this parameter, then
      ``os.path.join(paths[0], 'foo')``,
      ``os.path.join(paths[0], 'bar')``, and so on are added to the returned
      paths. Globs are accepted in this list and resolved using the function
      :py:func:`glob.glob`.
  suffix : :obj:`str`, optional
      ``suffix`` will be appended to all paths except ``prefixes``.

  Returns
  -------
  paths : [str]
      A list of unique and existing paths to be used in your search.
  """
  search = []
  suffix = suffix or ''

  # Priority 1: the environment
  if 'BOB_PREFIX_PATH' in os.environ:
    paths = os.environ['BOB_PREFIX_PATH'].split(os.pathsep)
    search += [p + suffix for p in paths]

  # Priority 2: user passed paths
  if prefixes:
    search += prefixes

  # Priority 3: the current system executable
  search.append(os.path.dirname(os.path.dirname(sys.executable)) + suffix)

  # Priority 4: the conda prefix
  conda_prefix = os.environ.get('CONDA_PREFIX')
  if conda_prefix:
    search.append(conda_prefix + suffix)

  # Priority 5: the default search prefixes
  search += [p + suffix for p in DEFAULT_PREFIXES]

  # Make unique to avoid searching twice
  search = uniq_paths(search)

  # Exhaustive combination of paths and subpaths
  if subpaths:
    subsearch = []
    for s in search:
      for p in subpaths:
        subsearch.append(os.path.join(s, p))
      subsearch.append(s)
    search = subsearch

  # Before we do a file-system check, filter out the un-existing paths
  tmp = []
  for k in search:
    tmp += glob.glob(k)
  search = tmp

  return search


def find_file(name, subpaths=None, prefixes=None):
  """Finds a generic file on the file system. Returns all occurrences.

  This method will find all occurrences of a given name on the file system and
  will return them to the user. It uses :any:`construct_search_paths` to
  construct the candidate folders that file may exist in.

  Parameters
  ----------
  name : str
      The name of the file. For example, ``gcc``.
  subpaths : [:obj:`str`], optional
      See :any:`construct_search_paths`
  subpaths : :obj:`str`, optional
      See :any:`construct_search_paths`

  Returns
  -------
  [str]
      A list of filenames that exist on the filesystem, matching your
      description.
  """

  search = construct_search_paths(prefixes=prefixes, subpaths=subpaths)

  retval = []
  for path in search:
    candidate = os.path.join(path, name)
    if os.path.exists(candidate):
      retval.append(candidate)

  return retval


def find_header(name, subpaths=None, prefixes=None):
  """Finds a header file on the file system. Returns all candidates.

  This method will find all occurrences of a given name on the file system and
  will return them to the user. It uses :any:`construct_search_paths` to
  construct the candidate folders that header may exist in accounting
  automatically for typical header folder names.

  Parameters
  ----------
  name : str
      The name of the header file.
  subpaths : [:obj:`str`], optional
      See :any:`construct_search_paths`
  subpaths : :obj:`str`, optional
      See :any:`construct_search_paths`

  Returns
  -------
  [str]
      A list of filenames that exist on the filesystem, matching your
      description.
  """

  headerpaths = []

  # arm-based system (e.g. raspberry pi 32 or 64-bit)
  if platform.machine().startswith('arm'):
    headerpaths += [os.path.join('include', 'arm-linux-gnueabihf')]

  # else, consider it intel compatible
  elif platform.architecture()[0] == '32bit':
    headerpaths += [os.path.join('include', 'i386-linux-gnu')]
  else:
    headerpaths += [os.path.join('include', 'x86_64-linux-gnu')]

  # Raspberry PI search directory (arch independent) + normal include
  headerpaths += ['include']

  # Exhaustive combination of paths and subpaths
  if subpaths:
    my_subpaths = []
    for hp in headerpaths:
      my_subpaths += [os.path.join(hp, k) for k in subpaths]
  else:
    my_subpaths = headerpaths

  return find_file(name, my_subpaths, prefixes)


def find_library(name, version=None, subpaths=None, prefixes=None,
    only_static=False):
  """Finds a library file on the file system. Returns all candidates.

  This method will find all occurrences of a given name on the file system and
  will return them to the user. It uses :any:`construct_search_paths` to
  construct the candidate folders that the library may exist in accounting
  automatically for typical library folder names.

  Parameters
  ----------
  name : str
      The name of the module to be found. If you'd like to find libz.so, for
      example, specify ``"z"``. For libmath.so, specify ``"math"``.
  version : :obj:`str`, optional
      The version of the library we are searching for. If not specified, then
      look only for the default names, such as ``libz.so`` and the such.
  subpaths : [:obj:`str`], optional
      See :any:`construct_search_paths`
  subpaths : :obj:`str`, optional
      See :any:`construct_search_paths`
  only_static : :obj:`bool`, optional
      A boolean, indicating if we should try only to search for static versions
      of the libraries. If not set, any would do.

  Returns
  -------
  [str]
      A list of filenames that exist on the filesystem, matching your
      description.
  """

  libpaths = []

  # arm-based system (e.g. raspberry pi 32 or 64-bit)
  if platform.machine().startswith('arm'):
    libpaths += [os.path.join('lib', 'arm-linux-gnueabihf')]

  # else, consider it intel compatible
  elif platform.architecture()[0] == '32bit':
    libpaths += [
        os.path.join('lib', 'i386-linux-gnu'),
        os.path.join('lib32'),
        ]
  else:
    libpaths += [
        os.path.join('lib', 'x86_64-linux-gnu'),
        os.path.join('lib64'),
        ]

  libpaths += ['lib']

  # Exhaustive combination of paths and subpaths
  if subpaths:
    my_subpaths = []
    for lp in libpaths:
      my_subpaths += [os.path.join(lp, k) for k in subpaths]
  else:
    my_subpaths = libpaths

  # Extensions to consider
  if only_static:
    extensions = ['.a']
  else:
    if sys.platform == 'darwin':
      extensions = ['.dylib', '.a']
    elif sys.platform == 'win32':
      extensions = ['.dll', '.a']
    else: # linux like
      extensions = ['.so', '.a']

  # The module names can be set with or without version number
  retval = []
  if version:
    for ext in extensions:
      if sys.platform == 'darwin': # version in the middle
        libname = 'lib' + name + '.' + version + ext
      else: # version at the end
        libname = 'lib' + name + ext + '.' + version

      retval += find_file(libname, my_subpaths, prefixes)

  for ext in extensions:
    libname = 'lib' + name + ext
    retval += find_file(libname, my_subpaths, prefixes)

  return retval

def find_executable(name, subpaths=None, prefixes=None):
  """Finds an executable on the file system. Returns all candidates.

  This method will find all occurrences of a given name on the file system and
  will return them to the user. It uses :any:`construct_search_paths` to
  construct the candidate folders that the executable may exist in accounting
  automatically for typical executable folder names.

  Parameters
  ----------
  name : str
      The name of the file. For example, ``gcc``.
  subpaths : [:obj:`str`], optional
      See :any:`construct_search_paths`
  prefixes : :obj:`str`, optional
      See :any:`construct_search_paths`

  Returns
  -------
  [str]
      A list of filenames that exist on the filesystem, matching your
      description.
  """

  binpaths = []

  # arm-based system (e.g. raspberry pi 32 or 64-bit)
  if platform.machine().startswith('arm'):
    binpaths += [os.path.join('bin', 'arm-linux-gnueabihf')]

  # else, consider it intel compatible
  elif platform.architecture()[0] == '32bit':
    binpaths += [
        os.path.join('bin', 'i386-linux-gnu'),
        os.path.join('bin32'),
        ]
  else:
    binpaths += [
        os.path.join('bin', 'x86_64-linux-gnu'),
        os.path.join('bin64'),
        ]

  binpaths += ['bin']

  # Exhaustive combination of paths and subpaths
  if subpaths:
    my_subpaths = []
    for lp in binpaths:
      my_subpaths += [os.path.join(lp, k) for k in subpaths]
  else:
    my_subpaths = binpaths

  # if conda-build's BUILD_PREFIX is set, use it as it may contain build tools
  # which are not available on the host environment
  prefixes = prefixes if prefixes is not None else []
  if 'BUILD_PREFIX' in os.environ:
    prefixes += [os.environ['BUILD_PREFIX']]

  # The module names can be set with or without version number
  return find_file(name, my_subpaths, prefixes)

def uniq(seq, idfun=None):
  """Very fast, order preserving uniq function"""

  # order preserving
  if idfun is None:
      def idfun(x): return x
  seen = {}
  result = []
  for item in seq:
      marker = idfun(item)
      # in old Python versions:
      # if seen.has_key(marker)
      # but in new ones:
      if marker in seen: continue
      seen[marker] = 1
      result.append(item)
  return result

def uniq_paths(seq):
  """Uniq'fy a list of paths taking into consideration their real paths"""
  return uniq([os.path.realpath(k) for k in seq if os.path.exists(k)])

def egrep(filename, expression):
  """Runs grep for a given expression on each line of the file

  Parameters:

  filename, str
    The name of the file to grep for the expression

  expression
    A regular expression, that will be initialized using :py:func:`re.compile`.

  Returns a list of re matches.
  """

  retval = []

  with open(filename, 'rt') as f:
    rexp = re.compile(expression)
    for line in f:
      p = rexp.match(line)
      if p: retval.append(p)

  return retval

def load_requirements(f=None):
  """Loads the contents of requirements.txt on the given path.

  Defaults to "./requirements.txt"
  """

  def readlines(f):
    retval = [str(k.strip()) for k in f]
    return [k for k in retval if k and k[0] not in ('#', '-')]

  # if f is None, use the default ('requirements.txt')
  if f is None:
    f = 'requirements.txt'
  if isinstance(f, str):
    f = open(f, 'rt')
  # read the contents
  return readlines(f)

def find_packages(directories=['bob']):
  """This function replaces the ``find_packages`` command from ``setuptools`` to search for packages only in the given directories.
  Using this function will increase the building speed, especially when you have (links to) deep non-code-related directory structures inside your package directory.
  The given ``directories`` should be a list of top-level sub-directories of your package, where package code can be found.
  By default, it uses ``'bob'`` as the only directory to search.
  """
  from setuptools import find_packages as _original
  if isinstance(directories, str):
    directories = [directories]
  packages = []
  for d in directories:
    packages += [d]
    packages += ["%s.%s" % (d, p) for p in _original(d)]
  return packages

def link_documentation(additional_packages = ['python', 'numpy'], requirements_file = "../requirements.txt", server = None):
  """Generates a list of documented packages on our documentation server for the packages read from the "requirements.txt" file and the given list of additional packages.

  Parameters:

  additional_packages : [str]
    A list of additional bob packages for which the documentation urls are added.
    By default, 'numpy' is added

  requirements_file : str or file-like
    The file (relative to the documentation directory), where to read the requirements from.
    If ``None``, it will be skipped.

  server : str or None
    The url to the server which provides the documentation.
    If ``None`` (the default), the ``BOB_DOCUMENTATION_SERVER`` environment variable is taken if existent.
    If neither ``server`` is specified, nor a ``BOB_DOCUMENTATION_SERVER`` environment variable is set, the default ``"http://www.idiap.ch/software/bob/docs/bob/%(name)s/%(version)s/"`` is used.

  """

  def smaller_than(v1, v2):
    """Compares scipy/numpy version numbers"""

    c1 = v1.split('.')
    c2 = v2.split('.')[:len(c1)] #clip to the compared version
    for i in range(len(c2)):
      n1 = c1[i]
      n2 = c2[i]
      try:
        n1 = int(n1)
        n2 = int(n2)
      except ValueError:
        n1 = str(n1)
        n2 = str(n2)
      if n1 < n2: return True
      if n1 > n2: return False
    return False


  if sys.version_info[0] <= 2:
    import urllib2 as urllib
    from urllib2 import HTTPError, URLError
  else:
    import urllib.request as urllib
    import urllib.error as error
    HTTPError = error.HTTPError
    URLError = error.URLError


  # collect packages are automatically included in the list of indexes
  packages = []
  version_re = re.compile(r'\s*[\<\>=]+\s*')
  if requirements_file is not None:
    if not isinstance(requirements_file, str) or \
        os.path.exists(requirements_file):
      requirements = load_requirements(requirements_file)
      packages += [version_re.split(k)[0] for k in requirements]
  packages += additional_packages


  def _add_index(name, addr, packages=packages):
    """Helper to add a new doc index to the intersphinx catalog

    Parameters:

      name (str): Name of the package that will be added to the catalog
      addr (str): The URL (except the ``objects.inv`` file), that will be added

    """

    if name in packages:
      print ("Adding intersphinx source for `%s': %s" % (name, addr))
      mapping[name] = (addr, None)
      packages = [k for k in packages if k != name]


  def _add_numpy_index():
    """Helper to add the numpy manual"""

    try:
      import numpy
      ver = numpy.version.version
      if smaller_than(ver, '1.5.z'):
        ver = '.'.join(ver.split('.')[:-1]) + '.x'
      else:
        ver = '.'.join(ver.split('.')[:-1]) + '.0'
      _add_index('numpy', 'https://docs.scipy.org/doc/numpy-%s/' % ver)

    except ImportError:
      _add_index('numpy', 'https://docs.scipy.org/doc/numpy/')


  def _add_scipy_index():
    """Helper to add the scipy manual"""

    try:
      import scipy
      ver = scipy.version.version
      if smaller_than(ver, '0.9.0'):
        ver = '.'.join(ver.split('.')[:-1]) + '.x'
      else:
        ver = '.'.join(ver.split('.')[:-1]) + '.0'
      _add_index('scipy', 'https://docs.scipy.org/doc/scipy-%s/reference/' % ver)

    except ImportError:
      _add_index('scipy', 'https://docs.scipy.org/doc/scipy/reference/')


  mapping = {}

  # add indexes for common packages used in Bob
  _add_index('python', 'https://docs.python.org/%d.%d/' % sys.version_info[:2])
  _add_numpy_index()
  _add_scipy_index()
  _add_index('matplotlib', 'http://matplotlib.org/')
  _add_index('setuptools', 'https://setuptools.readthedocs.io/en/latest/')
  _add_index('six', 'https://six.readthedocs.io')
  _add_index('sqlalchemy', 'https://docs.sqlalchemy.org/en/latest/')
  _add_index('docopt', 'http://docopt.readthedocs.io/en/latest/')
  _add_index('scikit-image', 'http://scikit-image.org/docs/dev/')
  _add_index('pillow', 'http://pillow.readthedocs.io/en/latest/')
  _add_index('click', 'http://click.pocoo.org/')


  # get the server for the other packages
  if server is None:
    if "BOB_DOCUMENTATION_SERVER" in os.environ:
      server = os.environ["BOB_DOCUMENTATION_SERVER"]
    else:
      server = "http://www.idiap.ch/software/bob/docs/bob/%(name)s/%(version)s/|http://www.idiap.ch/software/bob/docs/bob/%(name)s/master/"

  # array support for BOB_DOCUMENTATION_SERVER
  # transforms "(file:///path/to/dir  https://example.com/dir| http://bla )"
  # into ["file:///path/to/dir", "https://example.com/dir", "http://bla"]
  # so, trim eventual parenthesis/white-spaces and splits by white space or |
  if server.strip():
    server = re.split(r'[|\s]+', server.strip('() '))
  else:
    server = []

  # check if the packages have documentation on the server
  for p in packages:
    if p in mapping: continue #do not add twice...

    for s in server:
      # generate URL
      package_name = p.split()[0]
      if s.count('%s') == 1: #old style
        url = s % package_name
      else: #use new style, with mapping, try to link against specific version
        try:
          version = 'v' + pkg_resources.require(package_name)[0].version
        except pkg_resources.DistributionNotFound:
          version = 'stable' #package is not a runtime dep, only referenced
        url = s % {'name': package_name, 'version': version}

      try:
        # otherwise, urlopen will fail
        if url.startswith('file://'):
          f = urllib.urlopen(urllib.Request(url + 'objects.inv'))
          url = url[7:] #intersphinx does not like file://
        else:
          f = urllib.urlopen(urllib.Request(url))

        # request url
        print("Found documentation for %s on %s; adding intersphinx source" % (p, url))
        mapping[p] = (url, None)
        break #inner loop, for server, as we found a candidate!

      except HTTPError as exc:
        if exc.code != 404:
          # url request failed with a something else than 404 Error
          print("Requesting URL %s returned error: %s" % (url, exc))
          # notice mapping is not updated here, as the URL does not exist

      except URLError as exc:
        print("Requesting URL %s did not succeed (maybe offline?). " \
            "The error was: %s" % (url, exc))

      except IOError as exc:
        print ("Path %s does not exist. The error was: %s" % (url, exc))

  return mapping
