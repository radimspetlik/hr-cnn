#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import os
import warnings

from . import utils

from .file import File
from .utils import check_parameters_for_validity, \
    check_parameter_for_validity, \
    convert_names_to_highlevel, \
    convert_names_to_lowlevel, \
    file_names, \
    sort_files


class FileDatabase(object):
  """Low-level File-based Database API to be used within Bob.

  Not all Databases in Bob need to inherit from this class. Use this class
  only if in your database one sample correlates to one actual file.

  Attributes
  ----------
  original_directory : str
      The directory where the raw files are located.
  original_extension : str
      The extension of raw data files, e.g. ``.png``.
  """

  def __init__(self, original_directory, original_extension):
    self.original_directory = original_directory
    self.original_extension = original_extension

  def original_file_names(self, files):
    """Returns the full path of the original data of the given File objects.

    Parameters
    ----------
    files : list of :py:class:`bob.db.base.File`
        The list of file object to retrieve the original data file names for.

    Returns
    -------
    list of :obj:`str`
        The paths extracted for the files, in the same order.

    Raises
    ------
    ValueError
        if original_directory or original_extension is None
    """
    if self.original_directory is None:
      raise ValueError(
          'self.original_directory was not provided (must not be None)!')
    if self.original_extension is None:
      raise ValueError(
          'self.original_extension was not provided (must not be None)!')
    return self.file_names(
        files, self.original_directory, self.original_extension)

  def original_file_name(self, file):
    """This function returns the original file name for the given File
    object.

    Parameters
    ----------
    file
        :py:class:`bob.db.base.File` or a derivative
        The File objects for which the file name should be retrieved

    Returns
    -------
    str
        The original file name for the given :py:class:`bob.db.base.File`
        object.

    Raises
    ------
    ValueError
        if original_directory or original_extension is None.
    """
    # check if directory is set
    if not self.original_directory or not self.original_extension:
      raise ValueError(
          "The original_directory and/or the original_extension were not"
          " specified in the constructor.")
    # extract file name
    file_name = file.make_path(
        self.original_directory, self.original_extension)
    if not self.check_existence or os.path.exists(file_name):
      return file_name
    raise ValueError("The file '%s' was not found. Please check the "
                     "original directory '%s' and extension '%s'?" % (
                         file_name,
                         self.original_directory,
                         self.original_extension))

  # Deprecated Methods below

  def check_parameters_for_validity(self, parameters, parameter_description,
                                    valid_parameters,
                                    default_parameters=None):
    DeprecationWarning("This method is deprecated. Please use the "
                       "equivalent function in bob.db.base.utils")
    return check_parameters_for_validity(parameters, parameter_description,
                                         valid_parameters,
                                         default_parameters)

  def check_parameter_for_validity(self, parameter, parameter_description,
                                   valid_parameters, default_parameter=None):
    DeprecationWarning("This method is deprecated. Please use the "
                       "equivalent function in bob.db.base.utils")
    return check_parameter_for_validity(parameter, parameter_description,
                                        valid_parameters,
                                        default_parameter)

  def convert_names_to_highlevel(self, names, low_level_names,
                                 high_level_names):
    DeprecationWarning("This method is deprecated. Please use the "
                       "equivalent function in bob.db.base.utils")
    return convert_names_to_highlevel(names, low_level_names,
                                      high_level_names)

  def convert_names_to_lowlevel(self, names, low_level_names,
                                high_level_names):
    DeprecationWarning("This method is deprecated. Please use the "
                       "equivalent function in bob.db.base.utils")
    return convert_names_to_lowlevel(names, low_level_names,
                                     high_level_names)

  def file_names(self, files, directory, extension):
    DeprecationWarning("This method is deprecated. Please use the "
                       "equivalent function in bob.db.base.utils")
    return file_names(files, directory, extension)

  def sort(self, files):
    DeprecationWarning("This method is deprecated. Please use the "
                       "equivalent function (sort_files) in bob.db.base.utils")
    return sort_files(files)


class Database(FileDatabase):
  """This class is deprecated. New databases should use the
  :py:class:`bob.db.base.FileDatabase` class if required"""

  def __init__(self, original_directory=None, original_extension=None):
    warnings.warn("The bob.db.base.Database class is deprecated. "
                  "Please use bob.db.base.FileDatabase instead.",
                  DeprecationWarning)
    super(Database, self).__init__(original_directory, original_extension)


class SQLiteBaseDatabase(object):
  """This class can be used for handling SQL databases.

  It opens an SQL database in a read-only mode and keeps it opened during the
  whole session.


  Parameters
  ----------
  sqlite_file : str
      The file name (including full path) of the SQLite file to read or
      generate.

  file_class : :py:class:`bob.db.base.File`
      The ``File`` class, which needs to be derived from
      :py:class:`bob.db.base.File`. This is required to be able to
      :py:meth:`query` the databases later on.

  Attributes
  ----------
  m_file_class : :py:class:`bob.db.base.File`
      The `file_class` parameter is kept in this attribute.
  m_session : object
      The SQL session object.
  m_sqlite_file : str
      The `sqlite_file` parameter is kept in this attribute.
  """

  def __init__(self, sqlite_file, file_class):
    self.m_sqlite_file = sqlite_file
    if not os.path.exists(sqlite_file):
      self.m_session = None
    else:
      self.m_session = utils.session_try_readonly('sqlite', sqlite_file)

    # assert the given file class is derived from the File class
    assert issubclass(file_class, File)
    self.m_file_class = file_class

  def __del__(self):
    """Closes the connection to the database."""

    if self.is_valid():
      # do some magic to close the connection to the database file
      try:
        # Since the dispose function re-creates a pool
        # which might fail in some conditions, e.g., when this
        # destructor is called during the exit of the python
        # interpreter
        self.m_session.close()
        self.m_session.bind.dispose()
      except (TypeError, AttributeError, KeyError):
        # ... I can just ignore the according exception...
        pass

  def is_valid(self):
    """Returns if a valid session has been opened for reading the database.
    """

    return self.m_session is not None

  def assert_validity(self):
    """Raise a RuntimeError if the database back-end is not available."""

    if not self.is_valid():
      raise IOError(
          "Database of type 'sqlite' cannot be found at expected "
          "location '%s'." % self.m_sqlite_file)

  def query(self, *args):
    """Creates a query to the database using the given arguments."""

    self.assert_validity()
    return self.m_session.query(*args)

  def files(self, ids, preserve_order=True):
    """Returns a list of ``File`` objects with the given file ids

    Parameters
    ----------
    ids : :obj:`list` or :obj:`tuple`
        The ids of the object in the database table "file". This object
        should be a python iterable (such as a tuple or list).

    preserve_order : bool
        If True (the default) the order of elements is preserved, but the
        execution time increases.

    Returns
    -------
    list
        a list (that may be empty) of ``File`` objects.

    """

    file_objects = self.query(self.m_file_class).filter(
        self.m_file_class.id.in_(ids))
    if not preserve_order:
      return list(file_objects)
    else:
      path_dict = {}
      for f in file_objects:
        path_dict[f.id] = f
      return [path_dict[id] for id in ids]

  def paths(self, ids, prefix=None, suffix=None, preserve_order=True):
    """Returns a full file paths considering particular file ids


    Parameters
    ----------
    ids : :obj:`list` or :obj`tuple`
        The ids of the object in the database table "file". This object should
        be a python iterable (such as a tuple or list).
    prefix : :obj:`str`, optional
        The bit of path to be prepended to the filename stem
    suffix : :obj:`str`, optional
        The extension determines the suffix that will be appended to the
        filename stem.
    preserve_order : bool
        If True (the default) the order of elements is preserved, but the
        execution time increases.

    Returns
    -------
    list
        A list (that may be empty) of the fully constructed paths given
        the file ids.

    """

    file_objects = self.files(ids, preserve_order)
    return [f.make_path(prefix, suffix) for f in file_objects]

  def reverse(self, paths, preserve_order=True):
    """Reverses the lookup from certain paths, returns a list of
    :py:class:`bob.db.base.File`'s

    Parameters
    ----------
    paths : list
        The filename stems (list of str) to query for. This object should be a
        python iterable (such as a tuple or list)

    preserve_order : :obj:`bool`, optional
        If True (the default) the order of elements is preserved, but the
        execution time increases.

    Returns
    -------
    list
        A list (that may be empty).

    """

    file_objects = self.query(self.m_file_class).filter(
        self.m_file_class.path.in_(paths))
    if not preserve_order:
      return file_objects
    else:
      # path_dict = {f.path : f for f in file_objects}  <<-- works fine
      # with python 2.7, but not in 2.6
      path_dict = {}
      for f in file_objects:
        path_dict[f.path] = f
      return [path_dict[path] for path in paths]

  def uniquify(self, file_list):
    """Sorts the given list of File objects and removes duplicates from it.

    Parameters
    ----------
    file_list : [:py:class:`bob.db.base.File`]
        A list of File objects to be handled. Also other objects can be
        handled, as long as they are sortable.

    Returns
    -------
    list
        A sorted copy of the given ``file_list`` with the duplicates removed.
    """

    return sorted(set(file_list))

  def all_files(self, **kwargs):
    """Returns the list of all File objects that satisfy your query.

    For possible keyword arguments, please check the implemention's
    ``objects()`` method.
    """

    return self.uniquify(self.objects(**kwargs))


class SQLiteDatabase(SQLiteBaseDatabase, FileDatabase):
  """This class can be used for handling SQL **File** based databases.

  It inherits from :py:class:`bob.db.base.SQLiteBaseDatabase` and
  :py:class:`bob.db.base.FileDatabase`.

  """

  def __init__(self, sqlite_file, file_class,
               original_directory, original_extension):
    SQLiteBaseDatabase.__init__(self, sqlite_file, file_class)
    FileDatabase.__init__(self, original_directory, original_extension)
