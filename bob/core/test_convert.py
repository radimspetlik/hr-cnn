#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 18 Oct 13:50:08 2013

"""Tests for core conversion functions
"""

from . import convert
import numpy

def test_default_ranges():

  x = numpy.array([0,255,0,255,0,255], 'uint8').reshape(2,3)
  c = convert(x, 'uint16')
  assert numpy.array_equal(65535*(x.astype('uint16')/255), c)

def test_from_range():

  x = numpy.array([0, 0.2, 0.4, 0.6, 0.8, 1.0], 'float64')
  c = convert(x, 'uint8', source_range=(0,1))
  assert numpy.array_equal(c, (255*x).astype('uint8'))

def test_to_range():

  x = numpy.array(range(6), 'uint8').reshape(2,3)
  c = convert(x, 'float64', dest_range=(0.,255.))
  assert numpy.array_equal(x.astype('float64'), c)

def test_from_and_to_range():

  x = numpy.array(range(6), 'uint8').reshape(2,3)
  c = convert(x, 'float64', source_range=(0,255), dest_range=(0.,255.))
  assert numpy.array_equal(x.astype('float64'), c)


def test_sorting():
  # tests the sorting functionality implemented in C++
  from ._convert import _sort
  unsorted = numpy.random.random((100))
  py = numpy.sort(unsorted)
  _sort(unsorted)
  assert (py==unsorted).all()
