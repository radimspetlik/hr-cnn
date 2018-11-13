#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

'''This file defines all objects available in the database.

It must provide a confortable interface so that users can browse the database
contents in a nice, pythonic way, without worrying about file locations and
I/O.
'''

import os
from bob.db.base import File
import bob.io.base
import bob.io.image  # required because our database has images


class Sample(File):
  '''Defines a sample (image + tags) pair available in the database

  For file-based databases, you may inherit from :py:class:`bob.db.base.File`,
  which provides stock file loading/saving routines.

  Internally, a sample is composed of a root directory, pointing to where the
  database is installed, together with the file stem, indicating the common
  part of the name shared between the image and the tag annotation file.


  Parameters
  ----------
  data_dir : str
      The base directory where the root of the database is
      located on the user filesystem

  path : str
      The relative path (minus the extension) of the sample

  Attributes
  ----------
  data_dir : str
      The base directory where the root of the database is
      located on the user filesystem

  '''

  def __init__(self, data_dir, path):

    unique_id = os.path.basename(path)
    super(Sample, self).__init__(path, unique_id)
    self.data_dir = data_dir

  def make_path(self, directory=None, extension=None):
    '''Path construction routine - see :py:meth:`bob.db.base.File.make_path`'''

    extension = extension if extension is not None else '.png'
    directory = directory if directory is not None else self.data_dir
    return super(Sample, self).make_path(directory, extension)

  def load(self, directory=None, extension=None):
    '''Default loading routine - see :py:meth:`bob.db.base.File.load`'''

    path = self.make_path(directory, extension)
    return bob.io.base.load(path)

  @property
  def tags(self):
    '''A list of strings containing the tags for the image'''

    return [k.strip() for k in open(self.make_path(extension='.txt'), 'rt')
            if k.strip()]

  @property
  def dominant_color(self):
    '''The dominant color of the object'''
    return self.tags[1]

  def __repr__(self):
    return 'Sample("%s")' % self.path
