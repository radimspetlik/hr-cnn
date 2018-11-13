#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Sat  2 Nov 10:05:23 2013

"""Tests for bob.core.random
"""

from __future__ import division
from . import random
from .version import externals
import numpy
import nose.tools
import nose.plugins.skip
from distutils.version import LooseVersion
import functools
import math


def test_mt19937_creation():
  x = random.mt19937()
  y = random.mt19937()
  assert x == y


def test_mt19937_comparison():
  x = random.mt19937(10)
  y = random.mt19937(11)
  assert x != y


def test_uniform_creation():
  x = random.uniform('int8')
  assert x.dtype == numpy.int8
  assert x.min == 0
  assert x.max == 9


def test_uniform_int8():
  x = random.uniform('uint8', min=0, max=7)
  assert x.dtype == numpy.uint8
  rng = random.mt19937()
  l = [x(rng) for k in range(10000)]
  assert min(l) == 0
  assert max(l) == 7
  assert abs(sum(l)/10000 - 3.5) < 0.1


def test_uniform_float64():
  x = random.uniform('float64', min=-1, max=+1)
  assert x.dtype == numpy.float64
  rng = random.mt19937()
  l = [x(rng) for k in range(10000)]
  assert min(l) >= -1.0
  assert max(l) < 1.0
  assert abs(sum(l)/10000) < 0.1


def test_uniform_bool():
  x = random.uniform(bool)
  assert x.min == False
  assert x.max == True
  rng = random.mt19937()
  l = [x(rng) for k in range(1000)]
  assert min(l) == False
  assert max(l) == True


@nose.tools.raises(ValueError)
def test_uniform_bool_raises():
  x = random.uniform(bool, True, True)


@nose.tools.raises(NotImplementedError)
def test_uniform_complex():
  x = random.uniform('complex64')


def test_mt19937_same_sequence():
  x = random.uniform('float64', min=-1, max=+1)
  rng1 = random.mt19937(17)
  rng2 = random.mt19937(17)
  check = [x(rng1) == x(rng2) for k in range(1000)]
  assert numpy.all(check)


def test_mt19937_different_sequences():
  x = random.uniform('float64', min=-1, max=+1)
  rng1 = random.mt19937(17)
  rng2 = random.mt19937(-3)
  check = [x(rng1) == x(rng2) for k in range(1000)]
  assert not numpy.all(check)


def test_variate_generator_1d():
  x = random.variate_generator(random.mt19937(), random.uniform('float32', min=0, max=2*math.pi))
  m = x(10)
  assert m.shape == (10,)
  assert m.dtype == numpy.float32


def test_variate_generator_2d():
  x = random.variate_generator(random.mt19937(), random.uniform('uint16', min=0, max=65535))
  m = x((10,10))
  assert m.shape == (10,10)
  assert m.dtype == numpy.uint16


def test_normal():
  x = random.variate_generator(random.mt19937(), random.normal('float64', mean=0.5, sigma=2.0))
  assert x.distribution.mean == 0.5
  assert x.distribution.sigma == 2.0
  m = x(10000)
  assert abs(m.mean() - 0.5) < 0.1
  assert abs(m.std() - 2.) < 0.1


def test_lognormal():
  x = random.variate_generator(random.mt19937(), random.lognormal('float64', mean=0.0, sigma=1.0))
  assert x.distribution.mean == 0.0
  assert x.distribution.sigma == 1.0
  m = x(10000)
  assert abs(m.mean() - 1.65) < 0.1


def test_gamma():
  x = random.variate_generator(random.mt19937(), random.gamma('float64', alpha=0.5))
  assert x.distribution.alpha == 0.5
  m = x(10000)
  assert abs(m.mean() - 0.49) < 0.1
  assert abs(m.std() - 0.70) < 0.1


def test_binomial():
  x = random.variate_generator(random.mt19937(), random.binomial('float64', t=3.0, p=0.1))
  assert x.distribution.t == 3.0
  assert x.distribution.p == 0.1
  m = x(10000)
  assert abs(m.mean() - 0.30) < 0.1
  assert abs(m.std() - 0.52) < 0.1


def test_discrete():
  probs = (0.5, 0.2, 0.3) # mean == 0*0.5 + 1*0.2 + 2*0.3 = 0.8

  d = random.discrete(int, probs)
  assert numpy.allclose(probs, d.probabilities)

  x = random.variate_generator(random.mt19937(), d)
  m = x(10000)
  assert abs(m.mean() - 0.8) < 0.1


def test_repr():
  x = random.uniform(float)
  repr(x)
  x = random.normal(float)
  repr(x)
  x = random.lognormal(float)
  repr(x)
  x = random.gamma(float)
  repr(x)
  x = random.binomial(float)
  repr(x)
  x = random.discrete(int, (0.1, 0.9))
  repr(x)


def test_consistent_uniform_uint32():
  x = random.variate_generator(random.mt19937(42), random.uniform('uint32'))
  expected = [3, 7, 9]
  values = x(3)
  assert numpy.allclose(expected, values), \
      '%s not close to %s' % (expected, values)


def test_consistent_uniform_float64():
  x = random.variate_generator(random.mt19937(42), random.uniform('float64'))
  expected = [0.37454011, 0.79654298, 0.95071431]
  values = x(3)
  assert numpy.allclose(expected, values), \
      '%s not close to %s' % (expected, values)


def test_consistent_normal_float64():
  x = random.variate_generator(random.mt19937(42), random.normal('float64'))
  # from Boost >= 1.56 (see: https://svn.boost.org/trac/boost/ticket/9513)
  #expected = [-1.25821243, 0.6063474,  -0.19636614] #from <  1.56
  expected = [-0.63871374, -0.83680813, -0.40056644] #from >= 1.56
  values = x(3)
  assert numpy.allclose(expected, values), \
      '%s not close to %s' % (expected, values)


def test_consistent_lognormal_float64():
  x = random.variate_generator(random.mt19937(42), random.lognormal('float64', mean=.5))
  # from Boost >= 1.56 (see: https://svn.boost.org/trac/boost/ticket/9513)
  #expected = [0.04531668, 0.48256876, 0.17429895] #from <  1.56, deprecated implementation
  #expected = [0.09944391, 0.07734563, 0.13452042] #from >= 1.56, deprecated implementation
  expected = [0.87047717,  0.71404583,  1.10454508] #from >= 1.56, new implementation
  values = x(3)
  assert numpy.allclose(expected, values), \
      '%s not close to %s' % (expected, values)


def test_consistent_gamma_float64():
  x = random.variate_generator(random.mt19937(42), random.gamma('float64'))
  # from Boost >= 1.64, the underlying implementation of
  # exponential_distribution (on which the gamma_distribution is based on)
  # changed
  #expected = [ 0.46926808, 1.59230052, 3.01012154]
  expected = [ 0.75272492, 1.01650773, 0.46642008]
  values = x(3)
  assert numpy.allclose(expected, values), \
      '%s not close to %s' % (expected, values)


def test_consistent_binomial_float64():
  x = random.variate_generator(random.mt19937(42), random.binomial('float64'))
  expected = [0, 1, 1, 0, 1, 1, 1, 1, 0, 0]
  values = x(10)
  assert numpy.allclose(expected, values), \
      '%s not close to %s' % (expected, values)


def test_consistent_discrete_uint64():
  try:
    x = random.variate_generator(random.mt19937(42),
        random.discrete('uint64', [0.3, 0.3, 0.4]))
    expected = [1, 2, 2, 1, 0, 0, 0, 2, 1, 2]
    values = x(10)
    assert numpy.allclose(expected, values), \
        '%s not close to %s' % (expected, values)
  except NotImplementedError:
    pass
