#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Inverts the list of floating point numbers given on command line
"""

from .._library import reverse

def main():
  """Main routine, called by the script that gets the configuration of bob.blitz"""

  import sys
  if len(sys.argv) == 1:
    print ("Usage: %s <numbers>\n" % sys.argv[0])
    return

  numbers = [float(n) for n in sys.argv[1:]]
  print (numbers)
  rev = reverse(numbers)

  print ("%s reversed is %s" % (numbers, rev))
