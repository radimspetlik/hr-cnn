#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 16 Oct 12:16:49 2013

"""Tests for pkgconfig
"""

import nose
from .pkgconfig import pkgconfig, version

test_package = 'blitz'
pkg_config_version = '0.0'

def test_pkgconfig_version():

  assert version()
  global pkg_config_version
  pkg_config_version = version()

def test_detect_ok():
  pkg = pkgconfig(test_package)
  nose.tools.eq_(pkg.name, test_package)
  assert pkg.version
  #print pkg.name, pkg.version

@nose.tools.raises(RuntimeError)
def test_detect_not_ok():
  pkg = pkgconfig('foobarfoo')

def test_include_directories():
  pkg = pkgconfig(test_package)
  obj = pkg.include_directories()
  assert isinstance(obj, list)
  #assert obj
  #for k in obj:
  #  assert k.find('-I') != 0
  #print obj

def test_cflags_other():
  pkg = pkgconfig('freetype2')
  obj = pkg.cflags_other()
  #print obj

  #assert obj['extra_compile_args']
  #assert isinstance(obj['extra_compile_args'], list)
  #assert isinstance(obj['extra_compile_args'][0], tuple)
 
  #assert obj['define_macros']
  #assert isinstance(obj['define_macros'], list)
  #assert isinstance(obj['define_macros'][0], tuple) 

  assert isinstance(obj, dict)

def test_libraries():
  pkg = pkgconfig(test_package)
  obj = pkg.libraries()
  assert isinstance(obj, list)
  assert obj
  for k in obj:
    assert k.find('-l') != 0
  #print obj

def test_library_directories():
  pkg = pkgconfig(test_package)
  obj = pkg.library_directories()
  assert isinstance(obj, list)
  #assert obj
  #for k in obj:
  #  assert k.find('-L') != 0
  #print obj

def test_extra_link_args():
  pkg = pkgconfig(test_package)
  obj = pkg.extra_link_args()
  assert isinstance(obj, list)
  #print obj

def skip_on_pkgconfig_lt_024(func):
  from nose.plugins.skip import SkipTest
  from distutils.version import LooseVersion
  pkg_config_has_variables = LooseVersion(version()) >= '0.24'
  def _():
    if not pkg_config_has_variables: raise SkipTest('pkg-config version (%s) does not support variable listing' % pkg_config_version)
  _.__name__ = func.__name__
  return _

@skip_on_pkgconfig_lt_024
def test_variable_names():
  pkg = pkgconfig(test_package)
  obj = pkg.variable_names()
  assert isinstance(obj, list)
  assert obj
  #print obj

@skip_on_pkgconfig_lt_024
def test_variable():
  pkg = pkgconfig(test_package)
  names = pkg.variable_names()
  assert isinstance(names, list)
  assert names
  v = pkg.variable(names[0])
  assert v
  #print v

def test_macros():
  pkg = pkgconfig(test_package)
  macros = pkg.package_macros()
  assert isinstance(macros, list)
  assert macros
  assert macros[0][0].find('HAVE_') == 0
  assert macros[0][1] == '1'
  assert macros[1][0].find('_VERSION') > 0
  assert macros[1][1].find('"') == 0
  assert macros[1][1].rfind('"') == (len(macros[1][1]) - 1)
