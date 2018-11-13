from ._library import __doc__, mt19937, uniform, normal, lognormal, gamma, binomial, discrete
from ..version import api as __api_version__
import numpy

class variate_generator:
  """A pure-python version of the boost::variate_generator<> class

  **Constructor Parameters:**

  ``engine`` : :py:class:`mt19937`
    An instance of the already initialized RNG you would like to use.

  ``distribution`` : one of the distributions defined in :py:mod:`bob.core.random`
    The distribution to respect when generating scalars using the engine.
    The distribution object should be previously initialized.
  """

  def __init__(self, engine, distribution):

    self.engine = engine
    self.distribution = distribution

  def seed(self, value):
    """Resets the seed of the ``variate_generator`` with an (int) value"""

    self.engine.seed(value)
    self.distribution.reset()

  def __call__(self, shape=None):
    """__call__(shape) -> number

    Generates one or more random values

    **Parameters:**

    ``shape`` : tuple or ``None``
      If given, a :py:class:`numpy.ndarray` with the given shape will be returned.
      If ``None`` (the default), only a single random number will be drawn.

    **Returns:**

    number : float or :py:class:`numpy.ndarray`
      The generated random number(s).
    """
    if shape is None:
      return self.distribution(self.engine)
    else:
      l = [self.distribution(self.engine) for k in range(numpy.prod(shape))]
      return numpy.array(l).reshape(shape)
