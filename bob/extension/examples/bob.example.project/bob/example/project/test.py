#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Test Units
"""

def test_version():
  from .script import version
  assert version.main() == 0
