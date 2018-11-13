#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 20 Mar 2014 12:43:48 CET

"""Tests that the examples work as expected
"""

import os
import sys
import nose.tools
import tempfile
import shutil
import subprocess
import pkg_resources


def _run(package, run_call):
  temp_dir = tempfile.mkdtemp(prefix="bob_test")
  package_dir = os.path.join(temp_dir, 'bob.example.{0}'.format(package))

  try:
    base_path = pkg_resources.resource_filename(__name__, os.path.join('examples', 'bob.example.{0}'.format(package)))
    shutil.copytree(base_path, package_dir, symlinks=False, ignore=None)

    def _join(*args):
      a = (package_dir,) + args
      return os.path.join(*a)

    def _bin(path):
      return _join('bin', path)

    # buildout
    subprocess.call(['buildout', 'buildout:prefer-final=false'], cwd=package_dir, shell=True)
    assert os.path.exists(_bin('python'))

    # nosetests
    subprocess.call(['python', _bin('nosetests'), '-sv'])

    # check that the call is working
    subprocess.call(['python', _bin(run_call[0])] + run_call[1:])

    subprocess.call(['python', _bin('sphinx-build'), _join('doc'), _join('sphinx')])
    assert os.path.exists(_join('sphinx', 'index.html'))

    subprocess.call(['python', _bin('python'), '-c', 'import pkg_resources; from bob.example.%s import get_config; print(get_config())'%package])

  finally:
    shutil.rmtree(temp_dir)


def test_project():
  # Tests that the bob.example.project works
  _run('project', ['bob_example_project_version.py'])


def test_extension():
  # Tests that the bob.example.extension compiles and works
  _run('extension', ['bob_example_extension_reverse.py', '1', '2', '3', '4', '5'])


def test_library():
  # Tests that the bob.example.library compiles and works
  _run('library', ['bob_example_library_reverse.py', '1', '2', '3', '4', '5'])
