#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

'''Functionality to implement python-based config file parsing and loading.
'''

import imp
import pkg_resources
import pkgutil
from os.path import isfile
import logging

logger = logging.getLogger(__name__)

LOADED_CONFIGS = []


def _load_context(path, mod):
  '''Loads the Python file as module, returns a resolved context

  This function is implemented in a way that is both Python 2 and Python 3
  compatible. It does not directly load the python file, but reads its contents
  in memory before Python-compiling it. It leaves no traces on the file system.

  Parameters
  ----------
  path : str
      The full path of the Python file to load the module contents
      from
  mod : module
      A preloaded module to use as context for the next module
      loading. You can create a new module using :py:mod:`imp` as in ``m =
      imp.new_module('name'); m.__dict__.update(ctxt)`` where ``ctxt`` is a
      python dictionary with string -> object values representing the contents
      of the module to be created.

  Returns
  -------
  mod : :any:`module`
      A python module with the fully resolved context
  '''

  # executes the module code on the context of previously imported modules
  exec(compile(open(path, "rb").read(), path, 'exec'), mod.__dict__)

  return mod


def _get_module_filename(module_name):
  """Resolves a module name to an actual Python file.

  Parameters
  ----------
  module_name : str
      The name of the module

  Returns
  -------
  str
      The Python files that corresponds to the module name.
  """
  loader = pkgutil.get_loader(module_name)
  if loader is None:
    return ''
  try:
    return loader.path
  except AttributeError:
    return loader.filename


def _resolve_entry_point_or_modules(paths, entry_point_group):
  """Resolves a mixture of paths, entry point names, and module names to just
  paths. For example paths can be:
  ``paths = ['/tmp/config.py', 'config1', 'bob.extension.config2']``.

  Parameters
  ----------
  paths : [str]
      An iterable strings that either point to actual files, are entry point
      names, or are module names.
  entry_point_group : str
      The entry point group name to search in entry points.

  Raises
  ------
  ValueError
      If one of the paths cannot be resolved to an actual path to a file.

  Returns
  -------
  paths : [str]
      The resolved paths pointing to existing files.
  names : [str]
      The valid python module names to bind each of the files to

  """

  entries = {e.name: e for e in
             pkg_resources.iter_entry_points(entry_point_group)}
  files = []
  names = []

  for i, path in enumerate(paths):

    old_path = path
    module_name = 'user_config'  # fixed module name for files with full paths

    # if it already points to a file
    if isfile(path):
      pass

    # If it is an entry point name, collect path and module name
    elif path in entries:
      module_name = entries[path].module_name
      path = _get_module_filename(module_name)
      if not isfile(path):
        raise ValueError(
            "The specified entry point: `{}' pointing to module: `{}' and "
            "resolved to: `{}' does not point to an existing "
            "file.".format(old_path, module_name, path))

    # If it is not a path nor an entry point name, it is a module name then?
    else:
      # if we have gotten here so far then path is the module_name.
      module_name = path
      path = _get_module_filename(path)
      if not isfile(path):
        raise ValueError(
            "The specified path: `{}' resolved to: `{}' is not a file, not a "
            "entry point name of `{}', nor a module name".format(
                old_path, path, entry_point_group or ''))

    files.append(path)
    names.append(module_name)

  return files, names


def load(paths, context=None, entry_point_group=None):
  '''Loads a set of configuration files, in sequence

  This method will load one or more configuration files. Every time a
  configuration file is loaded, the context (variables) loaded from the
  previous file is made available, so the new configuration file can override
  or modify this context.

  Parameters
  ----------
  paths : [str]
      A list or iterable containing paths (relative or absolute) of
      configuration files that need to be loaded in sequence. Each
      configuration file is loaded by creating/modifying the context generated
      after each file readout.
  context : :py:class:`dict`, optional
      If provided, start the readout of the first configuration file with the
      given context. Otherwise, create a new internal context.
  entry_point_group : :py:class:`str`, optional
      If provided, it will treat non-existing file paths as entry point names
      under the ``entry_point_group`` name.

  Returns
  -------
  mod : :any:`module`
      A module representing the resolved context, after loading the provided
      modules and resolving all variables.

  '''
  # resolve entry points to paths
  if entry_point_group is not None:
    paths, names = _resolve_entry_point_or_modules(paths, entry_point_group)
  else:
    names = len(paths) * ['user_config']

  ctxt = imp.new_module('initial_context')
  if context is not None:
    ctxt.__dict__.update(context)
  # Small gambiarra (https://www.urbandictionary.com/define.php?term=Gambiarra)
  # to avoid the garbage collector to collect some already imported modules.
  LOADED_CONFIGS.append(ctxt)

  # if no paths are provided, return context
  if not paths:
    return ctxt

  for k, n in zip(paths, names):
    logger.debug("Loading configuration file `%s'...", k)
    mod = imp.new_module(n)
    # remove the keys that might break the loading of the next config file.
    ctxt.__dict__.pop('__name__', None)
    ctxt.__dict__.pop('__package__', None)
    mod.__dict__.update(ctxt.__dict__)
    LOADED_CONFIGS.append(mod)
    ctxt = _load_context(k, mod)

  return mod


def mod_to_context(mod):
  """Converts the loaded module of :any:`load` to a dictionary context.
  This function removes all the variables that start and end with ``__``.

  Parameters
  ----------
  mod : object
      What is returned by :any:`load`

  Returns
  -------
  dict
      The context that was in ``mod``.
  """
  return {k: v for k, v in mod.__dict__.items()
          if not (k.startswith('__') and k.endswith('__'))}
