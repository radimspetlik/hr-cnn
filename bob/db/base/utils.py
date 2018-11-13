#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 12 May 08:33:24 2011

"""Some utilities shared by many of the databases.
"""

import os


class null(object):
  """A look-alike stream that discards the input"""

  def write(self, s):
    """Writes contents of string ``s`` on this stream"""

    pass

  def flush(self):
    """Flushes the stream"""

    pass


def apsw_is_available():
  """Checks lock-ability for SQLite on the current file system"""

  try:
    import apsw  # another python sqlite wrapper (maybe supports URIs)
  except ImportError:
    return False

  # if you got here, apsw is available, check we have matching versions w.r.t
  # the sqlit3 module
  import sqlite3

  if apsw.sqlitelibversion() != sqlite3.sqlite_version:
    return False

  # if you get to this point, all seems OK
  return True


class SQLiteConnector(object):
  '''An object that handles the connection to SQLite databases.

  Parameters
  ----------
  filename : str
      The name of the file containing the SQLite database

  readonly : bool
      Should I try and open the database in read-only mode?

  lock : str
      Any vfs name as output by apsw.vfsnames()

  '''

  @staticmethod
  def filesystem_is_lockable(database):
    """Checks if the filesystem is lockable"""
    from sqlite3 import connect

    # memorize if the database was already there
    old = os.path.exists(database)
    conn = connect(database)

    retval = True
    try:
      conn.execute('PRAGMA synchronous = OFF')
    except Exception:
      retval = False
    finally:
      if not old and os.path.exists(database):
        os.unlink(database)

    return retval

  APSW_IS_AVAILABLE = apsw_is_available()

  def __init__(self, filename, readonly=False, lock=None):

    self.readonly = readonly
    self.vfs = lock
    self.filename = filename
    self.lockable = SQLiteConnector.filesystem_is_lockable(self.filename)

    if (self.readonly or (self.vfs is not None)) and \
            not self.APSW_IS_AVAILABLE and not self.lockable:
      import warnings
      warnings.warn(
          'Got a request for an SQLite connection using APSW, but I cannot '
          'find an sqlite3-compatible installed version of that module (or '
          'the module is not installed at all). Furthermore, the place where '
          'the database is sitting ("%s") is on a filesystem that does **not**'
          ' seem to support locks. I\'m returning a stock connection and '
          'hopping for the best.' % (filename,))

  def __call__(self):

    from sqlite3 import connect

    if (self.readonly or (self.vfs is not None)) and self.APSW_IS_AVAILABLE:
      # and not self.lockable
      import apsw
      if self.readonly:
        flags = apsw.SQLITE_OPEN_READONLY  # 1
      else:
        flags = apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE  # 2|4
      apsw_con = apsw.Connection(self.filename, vfs=self.vfs, flags=flags)
      return connect(apsw_con)

    return connect(self.filename, check_same_thread=False)

  def create_engine(self, echo=False):
    """Returns an SQLAlchemy engine"""

    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool
    return create_engine('sqlite://',
                         creator=self,
                         echo=echo,
                         poolclass=NullPool)

  def session(self, echo=False):
    """Returns an SQLAlchemy session"""

    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=self.create_engine(echo))
    return Session()


def session(dbtype, dbfile, echo=False):
  """Creates a session to an SQLite database"""

  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker

  url = connection_string(dbtype, dbfile)
  engine = create_engine(url, echo=echo)
  Session = sessionmaker(bind=engine)
  return Session()


def session_try_readonly(dbtype, dbfile, echo=False):
  """Creates a read-only session to an SQLite database.

  If read-only sessions are not supported by the underlying sqlite3 python DB
  driver, then a normal session is returned. A warning is emitted in case the
  underlying filesystem does not support locking properly.


  Raises:

    NotImplementedError: if the dbtype is not supported.

  """

  if dbtype != 'sqlite':
    raise NotImplementedError(
        "Read-only sessions are only currently supported for SQLite databases")

  connector = SQLiteConnector(dbfile, readonly=True, lock='unix-none')
  return connector.session(echo=echo)


def create_engine_try_nolock(dbtype, dbfile, echo=False):
  """Creates an engine connected to an SQLite database with no locks.

  If engines without locks are not supported by the underlying sqlite3 python
  DB driver, then a normal engine is returned. A warning is emitted if the
  underlying filesystem does not support locking properly in this case.


  Raises:

    NotImplementedError: if the dbtype is not supported.

  """

  if dbtype != 'sqlite':
    raise NotImplementedError(
        "Unlocked engines are only currently supported for SQLite databases")

  connector = SQLiteConnector(dbfile, lock='unix-none')
  return connector.create_engine(echo=echo)


def session_try_nolock(dbtype, dbfile, echo=False):
  """Creates a session to an SQLite database with no locks.

  If sessions without locks are not supported by the underlying sqlite3 python
  DB driver, then a normal session is returned. A warning is emitted if the
  underlying filesystem does not support locking properly in this case.


  Raises:

    NotImplementedError: if the dbtype is not supported.

  """

  if dbtype != 'sqlite':
    raise NotImplementedError(
        "Unlocked sessions are only currently supported for SQLite databases")

  connector = SQLiteConnector(dbfile, lock='unix-none')
  return connector.session(echo=echo)


def connection_string(dbtype, dbfile, opts={}):
  """Returns a connection string for supported platforms

  Parameters
  ----------
  dbtype : str
      The type of database (only ``sqlite`` is supported for the time being)

  dbfile : str
      The location of the file to be used
  opts : :obj:`dict`, optional
      This is ignored.

  Returns
  -------
  object
      The url.

  """

  from sqlalchemy.engine.url import URL
  return URL(dbtype, database=dbfile)


def resolved(x):
  return os.path.realpath(os.path.abspath(x))


def safe_tarmembers(archive):
  """Gets a list of safe members to extract from a tar archive


  This list excludes:
    * Full paths outside the destination sandbox
    * Symbolic or hard links to outside the destination sandbox

  Notes
  -----
    Code came from a StackOverflow answer
    http://stackoverflow.com/questions/10060069

  Example
  -------
    Deploy it like this
    .. code-block:: python

       ar = tarfile.open("foo.tar")
       ar.extractall(path="./sandbox", members=safe_tarmembers(ar))
       ar.close()

  Parameters
  ----------
  archive : tarfile.TarFile
      An opened tar file for reading

  Yields
  ------
  list
      A list of :py:class:`tarfile.TarInfo` objects that satisfy the security
      criteria imposed by this function, as denoted above.
    """

  def _badpath(path, base):
    # os.path.join will ignore base if path is absolute
    return not resolved(os.path.join(base, path)).startswith(base)

  def _badlink(info, base):
    # Links are interpreted relative to the directory containing the link
    tip = resolved(os.path.join(base, os.path.dirname(info.name)))
    return _badpath(info.linkname, base=tip)

  base = resolved(".")

  for finfo in archive:
    if _badpath(finfo.name, base):
      print("not extracting `%s': illegal path" % (finfo.name,))
    elif finfo.islnk() and _badlink(finfo, base):
      print("not extracting `%s': hard link to `%s'" %
            (finfo.name, finfo.linkname))
    elif finfo.issym() and _badlink(finfo, base):
      print("not extracting `%s': symlink to `%s'" %
            (finfo.name, finfo.linkname))
    else:
      yield finfo


def check_parameters_for_validity(parameters, parameter_description,
                                  valid_parameters, default_parameters=None):
  """Checks the given parameters for validity.

  Checks a given parameter is in the set of valid parameters. It also
  assures that the parameters form a tuple or a list.  If parameters is
  'None' or empty, the default_parameters will be returned (if
  default_parameters is omitted, all valid_parameters are returned).

  This function will return a tuple or list of parameters, or raise a
  ValueError.


  Parameters
  ----------
  parameters : str or list of :obj:`str` or None
      The parameters to be checked. Might be a string, a list/tuple of
      strings, or None.

  parameter_description : str
      A short description of the parameter. This will be used to raise an
      exception in case the parameter is not valid.

  valid_parameters : list of :obj:`str`
      A list/tuple of valid values for the parameters.

  default_parameters : list of :obj:`str` or None
      The list/tuple of default parameters that will be returned in case
      parameters is None or empty. If omitted, all valid_parameters are used.

  Returns
  -------
  tuple
      A list or tuple contatining the valid parameters.

  Raises
  ------
  ValueError
      If some of the parameters are not valid.

  """

  if parameters is None:
      # parameters are not specified, i.e., 'None' or empty lists
    parameters = default_parameters if default_parameters is not None \
        else valid_parameters

  if not isinstance(parameters, (list, tuple, set)):
    # parameter is just a single element, not a tuple or list -> transform it
    # into a tuple
    parameters = (parameters,)

  # perform the checks
  for parameter in parameters:
    if parameter not in valid_parameters:
      raise ValueError(
          "Invalid %s '%s'. Valid values are %s, or lists/tuples of those" %
          (parameter_description, parameter, valid_parameters))

  # check passed, now return the list/tuple of parameters
  return parameters


def check_parameter_for_validity(parameter, parameter_description,
                                 valid_parameters, default_parameter=None):
  """Checks the given parameter for validity

  Ensures a given parameter is in the set of valid parameters. If the
  parameter is ``None`` or empty, the value in ``default_parameter`` will
  be returned, in case it is specified, otherwise a :py:exc:`ValueError`
  will be raised.

  This function will return the parameter after the check tuple or list
  of parameters, or raise a :py:exc:`ValueError`.

  Parameters
  ----------
  parameter : :obj:`str` or :obj:`None`
      The single parameter to be checked. Might be a string or None.

  parameter_description : str
      A short description of the parameter. This will be used to raise an
      exception in case the parameter is not valid.

  valid_parameters : list of :obj:`str`
      A list/tuple of valid values for the parameters.

  default_parameter : list of :obj:`str`, optional
      The default parameter that will be returned in case parameter is None or
      empty. If omitted and parameter is empty, a ValueError is raised.

  Returns
  -------
  str
      The validated parameter.

  Raises
  ------
  ValueError
      If the specified parameter is invalid.

  """

  if parameter is None:
    # parameter not specified ...
    if default_parameter is not None:
      # ... -> use default parameter
      parameter = default_parameter
    else:
      # ... -> raise an exception
      raise ValueError(
          "The %s has to be one of %s, it might not be 'None'." % (
              parameter_description, valid_parameters))

  if isinstance(parameter, (list, tuple, set)):
    # the parameter is in a list/tuple ...
    if len(parameter) > 1:
      raise ValueError(
          "The %s has to be one of %s, it might not be more than one "
          "(%s was given)." % (parameter_description,
                               valid_parameters, parameter))
    # ... -> we take the first one
    parameter = parameter[0]

  # perform the check
  if parameter not in valid_parameters:
    raise ValueError(
        "The given %s '%s' is not allowed. Please choose one of %s." % (
            parameter_description, parameter, valid_parameters))

  # tests passed -> return the parameter
  return parameter


def convert_names_to_highlevel(names, low_level_names,
                               high_level_names):
  """
  Converts group names from a low level to high level API

  This is useful for example when you want to return ``db.groups()`` for
  the :py:mod:`bob.bio.base`. Your instance of the database should
  already have ``low_level_names`` and ``high_level_names`` initialized.

  """

  if names is None:
    return None
  mapping = dict(zip(low_level_names, high_level_names))
  if isinstance(names, str):
    return mapping.get(names)
  return [mapping[g] for g in names]


def convert_names_to_lowlevel(names, low_level_names,
                              high_level_names):
  """ Same as :py:meth:`convert_names_to_highlevel` but on reverse """

  if names is None:
    return None
  mapping = dict(zip(high_level_names, low_level_names))
  if isinstance(names, str):
    return mapping.get(names)
  return [mapping[g] for g in names]


def file_names(files, directory, extension):
  """file_names(files, directory, extension) -> paths

  Returns the full path of the given File objects.

  Parameters
  ----------
  files : list of :py:class:`bob.db.base.File`
      The list of file object to retrieve the file names for.

  directory : str
      The base directory, where the files can be found.

  extension : str
      The file name extension to add to all files.

  Returns
  -------
  paths : list of :obj:`str`
      The paths extracted for the files, in the same order.
  """
  # return the paths of the files, do not remove duplicates
  return [f.make_path(directory, extension) for f in files]


def sort_files(files):
  """Returns a sorted version of the given list of File's (or other structures
  that define an 'id' data member). The files will be sorted according to their
  id, and duplicate entries will be removed.

  Parameters
  ----------
  files : list of :py:class:`bob.db.base.File`
      The list of files to be uniquified and sorted.

  Returns
  -------
  sorted : list of :py:class:`bob.db.base.File`
      The sorted list of files, with duplicate `BioFile.id`\s being removed.
  """
  # sort files using their sort function
  sorted_files = sorted(files)
  # remove duplicates
  return [f for i, f in enumerate(sorted_files) if
          not i or sorted_files[i - 1].id != f.id]
