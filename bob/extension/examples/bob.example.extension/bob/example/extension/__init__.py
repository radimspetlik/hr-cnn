# import all packages with direct dependencies
# (This will load their pure C++ libraries, if needed)
import bob.blitz
# ... in fact, bob.blitz does not have a C++ library and it would not be needed to import it here
# ... nevertheless, it stays here not to forget it!


# import the C++ function ``reverse`` from the library
from ._library import reverse

# import the ``version`` library as well
from . import version as _version
version = _version.module

def get_config():
  """Returns a string containing the configuration information.
  """

  import bob.extension
  return bob.extension.get_config(__name__, _version.externals)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
