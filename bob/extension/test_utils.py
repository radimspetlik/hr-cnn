#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 20 Mar 2014 12:43:48 CET

"""Tests for file search utilities
"""

import os
import sys
import nose.tools
import pkg_resources
from nose.plugins.skip import SkipTest
from .utils import uniq, egrep, find_file, find_header, find_library, \
    load_requirements, find_packages, link_documentation

def test_uniq():

  a = [1, 2, 3, 7, 3, 2]

  nose.tools.eq_(uniq(a), [1, 2, 3, 7])

def test_find_file():

  f = find_file('array.h', subpaths=[os.path.join('include', 'blitz')])

  assert f

  nose.tools.eq_(os.path.basename(f[0]), 'array.h')

def test_find_header():

  f1 = find_file('array.h', subpaths=[os.path.join('include', 'blitz')])

  assert f1

  nose.tools.eq_(os.path.basename(f1[0]), 'array.h')

  f2 = find_header(os.path.join('blitz', 'array.h'))

  nose.tools.eq_(os.path.basename(f2[0]), 'array.h')

  assert f2

  nose.tools.eq_(f1, f2)

def test_find_library():

  f = find_library('blitz')

  assert f

  assert len(f) >= 1

  for k in f:
    assert k.find('blitz') >= 0

def test_egrep():

  f = find_header('version.hpp', subpaths=['boost', 'boost?*'])

  assert f

  matches = egrep(f[0], r"^#\s*define\s+BOOST_VERSION\s+(\d+)\s*$")

  nose.tools.eq_(len(matches), 1)

def test_find_versioned_library():

  f = find_header('version.hpp', subpaths=['boost', 'boost?*'])

  assert f

  matches = egrep(f[0], r"^#\s*define\s+BOOST_VERSION\s+(\d+)\s*$")

  nose.tools.eq_(len(matches), 1)

  version_int = int(matches[0].group(1))
  version_tuple = (
      version_int // 100000,
      (version_int // 100) % 1000,
      version_int % 100,
      )
  version = '.'.join([str(k) for k in version_tuple])

  lib = find_library('boost_system', version=version)
  lib += find_library('boost_system-mt', version=version)

  assert len(lib) >= 1

  for k in lib:
    assert k.find('boost_system') >= 0

def test_requirement_readout():

  if sys.version_info[0] == 3:
    from io import StringIO as stringio
  else:
    from cStringIO import StringIO as stringio

  f = """ # this is my requirements file
package-a >= 0.42
package-b
package-c
#package-e #not to be included

package-z
--no-index
-e http://example.com/mypackage-1.0.4.zip
"""

  result = load_requirements(stringio(f))
  expected = ['package-a >= 0.42', 'package-b', 'package-c', 'package-z']
  nose.tools.eq_(result, expected)


def test_find_packages():
  # tests the find-packages command inside the bob.extension package

  basedir = pkg_resources.resource_filename('bob.extension', '..')
  packages = find_packages(os.path.abspath(basedir))

  site_packages = os.path.dirname(os.path.commonprefix(packages))
  packages = [os.path.relpath(k, site_packages) for k in packages]

  assert 'bob' in packages
  assert 'bob.extension' in packages
  assert 'bob.extension.scripts' in packages


def test_documentation_generation():
  if sys.version_info[0] == 3:
    from io import StringIO as stringio
  else:
    from cStringIO import StringIO as stringio

  f = """ # this is my requirements file
package-a >= 0.42
package-b
package-c
#package-e #not to be included
setuptools

package-z
--no-index
-e http://example.com/mypackage-1.0.4.zip
"""

  # keep the nose tests quiet
  _stdout = sys.stdout

  try:
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull

    # test NumPy and SciPy docs
    try:
      import numpy
      result = link_documentation(['numpy'], None)
      assert len(result) == 1
      key = list(result.keys())[0]
      assert 'numpy' in key
    except ImportError:
      pass

    try:
      import scipy
      result = link_documentation(['scipy'], None)
      assert len(result) == 1
      key = list(result.keys())[0]
      assert 'scipy' in key
    except ImportError:
      pass

    try:
      import matplotlib
      result = link_documentation(['matplotlib'], None)
      assert len(result) == 1
      key = list(result.keys())[0]
      assert 'matplotlib' in key
    except ImportError:
      pass

    # test pypi packages
    additional_packages = [
        'python',
        'matplotlib',
        'bob.extension',
        'gridtk',
        'other.bob.package',
        ]

    # test linkage to official documentation
    server = "http://www.idiap.ch/software/bob/docs/bob/%s/master/"
    os.environ["BOB_DOCUMENTATION_SERVER"] = server
    result = link_documentation(additional_packages, stringio(f))
    expected = [
        'https://docs.python.org/%d.%d/' % sys.version_info[:2],
        'http://matplotlib.org/',
        'https://setuptools.readthedocs.io/en/latest/',
        server % 'bob.extension',
        server % 'gridtk',
        ]
    result = [k[0] for k in result.values()]
    nose.tools.eq_(sorted(result), sorted(expected))

  finally:
    sys.stdout = _stdout


def test_get_config():
  # Test the generic get_config() function
  import bob.extension
  cfg = bob.extension.get_config()
  splits = cfg.split("\n")
  assert splits[0].startswith('bob.extension')
  assert splits[1].startswith("* Python dependencies")
  assert any([s.startswith("  - setuptools") for s in splits[2:]])

  cfg = bob.extension.get_config("setuptools", {'MyPackage' : {'my_dict' : 42}}, 0x0204)
  splits = cfg.split("\n")
  assert splits[0].startswith('setuptools')
  assert "api=0x0204" in splits[0]
  assert splits[1].startswith("* C/C++ dependencies")
  assert any([s.startswith("  - MyPackage") for s in splits[2:]])
