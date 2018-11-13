#!/usr/bin/env python
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 23 Jun 20:22:28 2011 CEST
# vim: set fileencoding=utf-8 :

"""The db package contains simplified APIs to access data for various databases
that can be used in Biometry, Machine Learning or Pattern Classification."""

import pkg_resources

from . import utils, driver

from .file import File
from .database import Database, SQLiteBaseDatabase, SQLiteDatabase, FileDatabase
from .annotations import read_annotation_file
__version__ = pkg_resources.require(__name__)[0].version


def get_config():
  """Returns a string containing the configuration information.
  """
  import bob.extension
  return bob.extension.get_config(__name__)


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
    File,
    Database,
    FileDatabase,
    SQLiteDatabase,
    SQLiteBaseDatabase,
    read_annotation_file,
    )
__all__ = [_ for _ in dir() if not _.startswith('_')]
