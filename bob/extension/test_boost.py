#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 20 Mar 2014 12:43:48 CET

"""Tests for boost configuration
"""

import os
import sys
import nose
from .boost import boost
from distutils.version import LooseVersion

def test_boost_version():

  b = boost('>= 1.30')
  assert LooseVersion(b.version) >= '1.30'

def test_boost_simple_modules():

  b = boost()
  directories, libname = b.libconfig(['system'])
  nose.tools.eq_(len(directories), 1)
  assert os.path.exists(directories[0])
  nose.tools.eq_(len(libname), 1)

def test_boost_python_modules():

  b = boost()
  directories, libname = b.libconfig(['python'])
  nose.tools.eq_(len(directories), 1)
  assert os.path.exists(directories[0])
  nose.tools.eq_(len(libname), 1)
  #assert libname[0].find('-py%d%d' % sys.version_info[:2]) >= 0

def test_boost_multiple_modules():

  b = boost()
  directories, libname = b.libconfig(['python', 'system'])
  nose.tools.eq_(len(directories), 1)
  assert os.path.exists(directories[0])
  assert libname
  nose.tools.eq_(len(libname), 2)
  #assert libname[0].find('-py%d%d' % sys.version_info[:2]) >= 0
  #assert libname[1].find('-py%d%d' % sys.version_info[:2]) < 0

def test_common_prefix():

  b = boost()
  directories, libname = b.libconfig(['python', 'system'])
  nose.tools.eq_(len(directories), 1)
  assert os.path.exists(directories[0])
  os.path.commonprefix([directories[0], b.include_directory])
  assert len(os.path.commonprefix([directories[0], b.include_directory])) > 1
