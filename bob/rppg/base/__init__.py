from . import script

from .utils import crop_face

from ..chrom.extract_utils import compute_mean_rgb 
from ..chrom.extract_utils import project_chrominance 

def get_config():
  """Returns a string containing the configuration information.
  """

  import bob.extension
  return bob.extension.get_config(__name__)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
