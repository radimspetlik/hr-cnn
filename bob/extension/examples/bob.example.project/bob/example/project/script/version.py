#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Prints the version of bob and exits
"""

def main():
  """Main routine, called by the script that gets the configuration of bob.blitz"""

  import bob.blitz
  print (bob.blitz.get_config())
  return 0

