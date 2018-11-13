#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 20 Mar 2014 12:43:48 CET

"""Tests for scripts
"""

import os
import sys
import nose.tools

def test_new_version():
  # Tests the bob_new_version.py script

  from bob.extension.scripts import new_version

  # test the script using the dry-run option (to avoid to actually tag and push code)

  # assert that it does not raise an exception, when both latest and stable version are specified
  if os.path.exists("version.txt"):
    new_version(['--dry-run', '--stable-version', '20.7.0', '--latest-version', '20.8.0'])

  # assert that it does not raise an exception, when only the latest version is specified
  if os.path.exists("version.txt"):
    new_version(['--dry-run', '--latest-version', '20.8.0'])

  # assert that no exception is raised even if no version is specified
  if os.path.exists("version.txt"):
    new_version(['--dry-run'])

  # assert that no exception is raised even if no version is specified
  if os.path.exists("version.txt"):
    new_version(['--dry-run', '--minor'])

  # assert that no exception is raised even if no version is specified
  if os.path.exists("version.txt"):
    new_version(['--dry-run', '--major'])

  # assert that it does raise an exception, when the --stable-version and
  # --minor and/or --major is provided
  nose.tools.assert_raises(ValueError, new_version, ['--dry-run', '--stable-version', '100.8.0', '--minor'])
  nose.tools.assert_raises(ValueError, new_version, ['--dry-run', '--stable-version', '100.8.0', '--major'])
  nose.tools.assert_raises(ValueError, new_version, ['--dry-run', '--stable-version', '100.8.0', '--minor', '--major'])

  # assert that it does raise an exception, when the latest version is too low
  nose.tools.assert_raises(ValueError, new_version, ['--dry-run', '--latest-version', '0.8.0'])

  # assert that it does raise an exception, when the stable version is too low
  nose.tools.assert_raises(ValueError, new_version, ['--dry-run', '--stable-version', '0.8.0', '--latest-version', '0.9.0'])

  # assert that it does raise an exception, when the stable version is higher than latest version
  nose.tools.assert_raises(ValueError, new_version, ['--dry-run', '--stable-version', '20.8.0', '--latest-version', '20.8.0a1'])

  # assert that it does not raise an exception, when --force is given and the latest version is too low
  if os.path.exists("version.txt"):
    new_version(['--force', '--dry-run', '--latest-version', '0.8.0'])

  # assert that it does not raise an exception, when --force is given and the stable version is too low
  if os.path.exists("version.txt"):
    new_version(['--force', '--dry-run', '--stable-version', '0.8.0', '--latest-version', '0.9.0'])

  # assert that it does not raise an exception, when --force is given and the stable version is higher than latest version
  if os.path.exists("version.txt"):
    new_version(['--force', '--dry-run', '--stable-version', '20.8.0', '--latest-version', '20.8.0a1'])
