#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

'''Typical database interface for a package.

Our object is to define a database class that will allow us to query for its
samples.
'''


# Typically, samples are defined in a separate "models.py" file
# containing all kinds of important "objects" available in the database
from . models import Sample


class Database:
  '''Sample database implementation. Documentation is **very** important.

  You should use sphinx-napoleon (numpy) to document parameters and methods
  '''

  def __init__(self):

    # in this example, the database location is fixed
    import pkg_resources
    self.data_dir = pkg_resources.resource_filename(__name__, 'data')

    # Because this database is small, we store all files we know about in this
    # list
    self.allfiles = [
        'dir1/sample1-1',
        'dir1/sample1-2',
        'dir2/sample2-1',
        'dir2/sample2-2',
        'dir3/sample3-1',
        'dir3/sample3-2',
    ]

  def objects(self, group=None):
    '''Provides an iterable over samples given the selector information

    This method returns an iterable (it may be a list, an iterator or a
    generator) allowing the user to iterate over the samples, given the
    selection criteria.

    The selection criteria is database dependent. Given the simple nature of
    our database, our selector allows only to subselect samples for a
    particular group given the design protocol.


    Parameters
    ----------
    group : str
        A string that defines the subset within the database, to
        return the iteratable for. It may take the value ``test`` or ``train``.


    Returns
    -------
    list
        A list of :py:class:`.Sample` objects you can use to create processing
        pipelines.

    Raises
    ------
    ValueError
        in case the supplied ``group`` value is not valid.

    '''

    if group is not None and group not in ('train', 'test'):
      raise ValueError('parameter "groups" should be one of "train" or "test"'
                       ' - the value "%s" is not valid' % group)

    # this is a very simple database, so we make simple code here as well
    # more sophisticated databases may need more intricate design however

    retlist = self.allfiles
    if group is not None:
      if group == 'train':
        retlist = [k for k in retlist if k.endswith('-1')]
      else:  # it is test
        retlist = [k for k in retlist if k.endswith('-2')]

    return [Sample(self.data_dir, k) for k in retlist]
