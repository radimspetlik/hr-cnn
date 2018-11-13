#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 20 Mar 2014 12:43:48 CET

"""Tests for file search utilities
"""

import os
import sys
import shutil
import nose.tools
import platform

import bob.extension
import pkg_resources

import tempfile

def _find(lines, start):
  for i, line in enumerate(lines):
    if line.find(start) == 0:
      return i
  assert False

def test_cmake_list():
  # checks that the CMakeLists.txt file is generated properly

  generator = bob.extension.CMakeListsGenerator(
    name = 'bob_cmake_test',
    sources = ['cmake_test.cpp'],
    target_directory = "test_target",
    version = '3.2.1',
    include_directories = ["/usr/include/test"],
    system_include_directories = ["/usr/include/test_system"],
    libraries = ['some_library'],
    library_directories = ["/usr/include/test"],
    macros = [("TEST", "MACRO")]
  )

  temp_dir = tempfile.mkdtemp(prefix="bob_extension_test_")

  generator.generate(temp_dir, temp_dir)

  # read created file
  lines = [line.rstrip() for line in open(os.path.join(temp_dir, "CMakeLists.txt"))]

  # check that all elements are properly written in the file
  assert lines[_find(lines, 'project')] == 'project(bob_cmake_test)'
  assert lines[_find(lines, 'include')] == 'include_directories(/usr/include/test)'
  assert lines[_find(lines, 'include_directories(SYSTEM')] == 'include_directories(SYSTEM /usr/include/test_system)'
  assert lines[_find(lines, 'link')] == 'link_directories(/usr/include/test)'
  assert lines[_find(lines, 'add')] == 'add_definitions(-DTEST=MACRO)'

  index = _find(lines, 'add_library')
  assert lines[index+1].find('cmake_test.cpp') >= 0

  index = _find(lines, 'set_target_properties')
  assert lines[index+1].find('test_target') >= 0

  assert lines[_find(lines, 'target_link_libraries')].find('some_library') >= 0

  # finally, clean up the mess
  shutil.rmtree(temp_dir)


def test_library():
  old_dir = os.getcwd()
  temp_dir = tempfile.mkdtemp(prefix="bob_extension_test_")
  target_dir = os.path.join(temp_dir, 'build', 'lib', 'target')
  # copy test file to temp directory
  shutil.copyfile(pkg_resources.resource_filename(__name__, 'test_documentation.cpp'), os.path.join(temp_dir, 'test_documentation.cpp'))
  os.chdir(temp_dir)
  # check that the library compiles and links properly
  library = bob.extension.Library(
    name = 'target.bob_cmake_test',
    sources = ['test_documentation.cpp'],
    include_dirs = [pkg_resources.resource_filename(__name__, 'include')],
    version = '3.2.1'
  )

  # redirect output of functions to /dev/null to avoid spamming the console
  devnull = open(os.devnull, 'w')
  # compile
  compile_dir = os.path.join(temp_dir, 'build', 'lib')
  os.makedirs(compile_dir)
  os.makedirs(target_dir)
  library.compile(compile_dir,stdout=devnull)

  # check that the library was generated sucessfully
  if platform.system() == 'Darwin':
    lib_name = 'libbob_cmake_test.dylib'
  else:
    lib_name = 'libbob_cmake_test.so'

  assert os.path.exists(os.path.join(target_dir, lib_name))

  os.chdir(old_dir)

  # TODO: compile a test executable to actually link the library

  # finally, clean up the mess
  shutil.rmtree(temp_dir)
