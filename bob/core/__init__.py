# import our own Library
import bob.extension
bob.extension.load_bob_library('bob.core', __file__)

from ._convert import convert
from . import log
from . import random
from . import version
from .version import module as __version__
from .version import api as __api_version__

def get_config():
  """Returns a string containing the configuration information.
  """
  return bob.extension.get_config(__name__, version.externals, version.api)


def get_macros():
  """get_macros() -> macros

  Returns a list of preprocessor macros, such as ``[(HAVE_BOOST, 1), (BOOST_VERSION,xx)]``.
  This function is automatically used by :py:func:`bob.extension.get_bob_libraries` to retrieve the prerpocessor definitions that are required to use the C bindings of this library in dependent classes.
  You shouldn't normally need to call this function by hand.

  **Returns:**

  ``macros`` : [str]
    The list of preprocessor macros required to use the C bindings of this class.
  """
  # try to use pkg_config first
  from bob.extension.boost import boost
  macros = []
  try:
    b = boost()
    macros = b.macros()
  except:
    pass
  return macros


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
