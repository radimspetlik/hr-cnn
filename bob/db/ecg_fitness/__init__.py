#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""A very simple API to query and load data from the COHFACE database.
"""

from models import *
from driver import Interface, DATABASE_LOCATION

from pkg_resources import resource_filename


class Database(object):

    def __init__(self, dbdir=DATABASE_LOCATION):
        self.info = Interface()

        self.metadata = []
        for path, dirs, files in os.walk(dbdir):
            if 'c920-1.avi' in files:
                relpath = os.path.relpath(path, dbdir)
                self.metadata.append(os.path.join(relpath, 'c920-1'))
            if 'c920-2.avi' in files:
                relpath = os.path.relpath(path, dbdir)
                self.metadata.append(os.path.join(relpath, 'c920-2'))

    def objects(self, protocol='all', subset=None):
        """Returns a list of unique :py:class:`.File` objects for the specific
        query by the user.


        Parameters:

          protocol (:py:class:`str`, optional): If set, it should be
            ``all``.  All, which is the default.

          subset (:py:class:`str`, optional): If set, it could be either ``train``
            or ``test`` or a combination of them (i.e. a list). If not set
            (default), the files from all these sets are retrieved, according to
            the protocol.

        Returns:

          list: A list of :py:class:`File` objects.

        """
        files = []

        proto_basedir = os.path.join('data', 'protocols')

        if not subset:
            d = resource_filename(__name__, os.path.join(proto_basedir, protocol, 'all.txt'))
            with open(d, 'rt') as f:
                sessions = f.read().split()
            files += [File(k) for k in self.metadata if k in sessions]
            return files
        else:
            if 'train' in subset:
                d = resource_filename(__name__, os.path.join(proto_basedir, protocol, 'train.txt'))
                with open(d, 'rt') as f: sessions = f.read().split()
                files += [File(k) for k in self.metadata if k in sessions]
            if 'test' in subset:
                d = resource_filename(__name__, os.path.join(proto_basedir, protocol, 'test.txt'))
                with open(d, 'rt') as f: sessions = f.read().split()
                files += [File(k) for k in self.metadata if k in sessions]
            if 'dev' in subset:
                d = resource_filename(__name__, os.path.join(proto_basedir, protocol, 'dev.txt'))
                with open(d, 'rt') as f: sessions = f.read().split()
                files += [File(k) for k in self.metadata if k in sessions]

        return files


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
)

__all__ = [_ for _ in dir() if not _.startswith('_')]
