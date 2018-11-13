#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

'''Minimal database implementation'''

from .models import Sample
from .query import Database


# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
  """Says object was actually declared here, an not on the import module.

  Parameters:

    *args: An iterable of objects to modify

  Resolves `Sphinx referencing issues
  <https://github.com/sphinx-doc/sphinx/issues/3048>`
  """

  for obj in args: obj.__module__ = __name__

__appropriate__(
    Sample,
    Database,
    )
__all__ = [_ for _ in dir() if not _.startswith('_')]
