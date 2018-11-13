#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Test Units
"""

def test_reverse():
  from . import reverse
  count = 10000
  source = [float(f) for f in range(count)]
  target = reverse(source)
  for i in range(count):
    assert target[i] == source[count-i-1]
