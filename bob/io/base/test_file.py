#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Nov 16 13:27:15 2011 +0100
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""A combined test for all built-in types of Array/interaction in
python.
"""

import os
import sys
import numpy
import nose.tools

from . import load, write, peek, peek_all, File, test_utils

def test_peek():

  f = test_utils.datafile('test1.hdf5', __name__)
  assert peek(f) == (numpy.uint16, (3,), (1,))
  assert peek_all(f) == (numpy.uint16, (3,3), (3,1))

def test_iteration():

  fname = test_utils.datafile('matlab_2d.hdf5', __name__)
  f = File(fname, 'r')
  nose.tools.eq_(len(f), 512)

  objs = load(fname)

  for l, i in zip(objs, f):
    assert numpy.allclose(l, i)

def test_indexing():

  fname = test_utils.datafile('matlab_2d.hdf5', __name__)
  f = File(fname, 'r')
  nose.tools.eq_(len(f), 512)

  objs = load(fname)
  nose.tools.eq_(len(f), len(objs))

  # simple indexing
  assert numpy.allclose(f[0], objs[0])
  assert numpy.allclose(f[1], objs[1])
  assert numpy.allclose(f[-1], objs[-1])
  assert numpy.allclose(f[-2], objs[-2])

def test_slicing_empty():

  fname = test_utils.datafile('matlab_2d.hdf5', __name__)
  f = File(fname, 'r')

  objs = f[1:1]
  assert objs.shape == tuple()

def test_slicing_0():

  fname = test_utils.datafile('matlab_2d.hdf5', __name__)
  f = File(fname, 'r')

  objs = f[:]
  for i, k in enumerate(load(fname)):
    assert numpy.allclose(k, objs[i])

def test_slicing_1():

  fname = test_utils.datafile('matlab_2d.hdf5', __name__)
  f = File(fname, 'r')

  # get slice
  s1 = f[3:10:2]
  nose.tools.eq_(len(s1), 4)
  assert numpy.allclose(s1[0], f[3])
  assert numpy.allclose(s1[1], f[5])
  assert numpy.allclose(s1[2], f[7])
  assert numpy.allclose(s1[3], f[9])

def test_slicing_2():

  fname = test_utils.datafile('matlab_2d.hdf5', __name__)
  f = File(fname, 'r')

  # get negative slicing
  s = f[-10:-2:3]
  nose.tools.eq_(len(s), 3)
  assert numpy.allclose(s[0], f[len(f)-10])
  assert numpy.allclose(s[1], f[len(f)-7])
  assert numpy.allclose(s[2], f[len(f)-4])

def test_slicing_3():

  fname = test_utils.datafile('matlab_2d.hdf5', __name__)
  f = File(fname, 'r')

  # get negative stepping slice
  s = f[20:10:-3]
  nose.tools.eq_(len(s), 4)
  assert numpy.allclose(s[0], f[20])
  assert numpy.allclose(s[1], f[17])
  assert numpy.allclose(s[2], f[14])
  assert numpy.allclose(s[3], f[11])

def test_slicing_4():

  fname = test_utils.datafile('matlab_2d.hdf5', __name__)
  f = File(fname, 'r')

  # get all negative slice
  s = f[-10:-20:-3]
  nose.tools.eq_(len(s), 4)
  assert numpy.allclose(s[0], f[len(f)-10])
  assert numpy.allclose(s[1], f[len(f)-13])
  assert numpy.allclose(s[2], f[len(f)-16])
  assert numpy.allclose(s[3], f[len(f)-19])

@nose.tools.raises(TypeError)
def test_indexing_type_check():

  f = File(test_utils.datafile('matlab_2d.hdf5', __name__), 'r')
  nose.tools.eq_(len(f), 512)
  f[4.5]

@nose.tools.raises(IndexError)
def test_indexing_boundaries():

  f = File(test_utils.datafile('matlab_2d.hdf5', __name__), 'r')
  nose.tools.eq_(len(f), 512)
  f[512]

@nose.tools.raises(IndexError)
def test_indexing_negative_boundaries():
  f = File(test_utils.datafile('matlab_2d.hdf5', __name__), 'r')
  nose.tools.eq_(len(f), 512)
  f[-513]

def transcode(filename):
  """Runs a complete transcoding test, to and from the binary format."""

  tmpname = test_utils.temporary_filename(suffix=os.path.splitext(filename)[1])

  try:
    # transcode from test format into the test format -- test array access modes
    orig_data = load(filename)
    write(orig_data, tmpname)
    rewritten_data = load(tmpname)

    assert numpy.array_equal(orig_data, rewritten_data)

    # transcode to test format -- test arrayset access modes
    trans_file = File(tmpname, 'w')
    index = [slice(orig_data.shape[k]) for k in range(len(orig_data.shape))]
    for k in range(orig_data.shape[0]):
      index[0] = k
      trans_file.append(orig_data[index]) #slice from first dimension
    del trans_file

    rewritten_file = File(tmpname, 'r')

    for k in range(orig_data.shape[0]):
      rewritten_data = rewritten_file.read(k)
      index[0] = k
      assert numpy.array_equal(orig_data[index], rewritten_data)

  finally:
    # And we erase both files after this
    if os.path.exists(tmpname): os.unlink(tmpname)

def array_readwrite(extension, arr, close=False):
  """Runs a read/write verify step using the given numpy data"""
  tmpname = test_utils.temporary_filename(suffix=extension)
  try:
    write(arr, tmpname)
    reloaded = load(tmpname)
    if close: assert numpy.allclose(arr, reloaded)
    else: assert numpy.array_equal(arr, reloaded)
  finally:
    if os.path.exists(tmpname): os.unlink(tmpname)

def arrayset_readwrite(extension, arrays, close=False):
  """Runs a read/write verify step using the given numpy data"""
  tmpname = test_utils.temporary_filename(suffix=extension)
  try:
    f = File(tmpname, 'w')
    for k in arrays:
      f.append(k)
    del f
    f = File(tmpname, 'r')
    for k, array in enumerate(arrays):
      reloaded = f.read(k) #read the contents
      if close:
        assert numpy.allclose(array, reloaded)
      else: assert numpy.array_equal(array, reloaded)
  finally:
    if os.path.exists(tmpname): os.unlink(tmpname)

def test_hdf5():

  # array writing tests
  a1 = numpy.random.normal(size=(2,3)).astype('float32')
  a2 = numpy.random.normal(size=(2,3,4)).astype('float64')
  a3 = numpy.random.normal(size=(2,3,4,5)).astype('complex128')
  a4 = (10 * numpy.random.normal(size=(3,3))).astype('uint64')

  array_readwrite('.hdf5', a1) # extensions: .hdf5 or .h5
  array_readwrite(".h5", a2)
  array_readwrite('.h5', a3)
  array_readwrite(".h5", a4)
  array_readwrite('.h5', a3[:,::2,::2,::2]) #test non-contiguous

  # arrayset writing tests
  a1 = []
  a2 = []
  a3 = []
  a4 = []
  for k in range(10):
    a1.append(numpy.random.normal(size=(2,3)).astype('float32'))
    a2.append(numpy.random.normal(size=(2,3,4)).astype('float64'))
    a3.append(numpy.random.normal(size=(2,3,4,5)).astype('complex128'))
    a4.append((10*numpy.random.normal(size=(3,3))).astype('uint64'))

  arrayset_readwrite('.h5', a1)
  arrayset_readwrite(".h5", a2)
  arrayset_readwrite('.h5', a3)
  arrayset_readwrite(".h5", a4)

  # complete transcoding tests
  transcode(test_utils.datafile('test1.hdf5', __name__))
  transcode(test_utils.datafile('matlab_1d.hdf5', __name__))
  transcode(test_utils.datafile('matlab_2d.hdf5', __name__))

@test_utils.extension_available('.bindata')
def test_torch3_binary():

  # array writing tests
  a1 = numpy.random.normal(size=(3,4)).astype('float32') #good, supported
  a2 = numpy.random.normal(size=(3,4)).astype('float64') #good, supported
  a3 = numpy.random.normal(size=(3,4)).astype('complex128') #not supported

  array_readwrite('.bindata', a1)
  array_readwrite(".bindata", a2)
  nose.tools.assert_raises(RuntimeError, array_readwrite, ".bindata", a3)

  # arrayset writing tests
  a1 = []
  a2 = []
  a3 = []
  a4 = []
  for k in range(10):
    a1.append(numpy.random.normal(size=(24,)).astype('float32')) #supported
    a2.append(numpy.random.normal(size=(24,)).astype('float64')) #supported
    a3.append(numpy.random.normal(size=(24,)).astype('complex128')) #unsupp.
    a4.append(numpy.random.normal(size=(3,3))) #not supported

  arrayset_readwrite('.bindata', a1)
  arrayset_readwrite(".bindata", a2)

  # checks we raise if we don't suppport a type
  nose.tools.assert_raises(RuntimeError, arrayset_readwrite, ".bindata", a3)
  nose.tools.assert_raises(RuntimeError, arrayset_readwrite, ".bindata", a4)

  # complete transcoding test
  transcode(test_utils.datafile('torch3.bindata', __name__))

@test_utils.extension_available('.tensor')
def test_tensorfile():

  # array writing tests
  a1 = numpy.random.normal(size=(3,4)).astype('float32')
  a2 = numpy.random.normal(size=(3,4,5)).astype('float64')
  a3 = (100*numpy.random.normal(size=(2,3,4,5))).astype('int32')

  array_readwrite('.tensor', a1)
  array_readwrite(".tensor", a2)
  array_readwrite(".tensor", a3)
  array_readwrite('.tensor', a3[::2,::2]) #test non-contiguous

  # arrayset writing tests
  a1 = []
  a2 = []
  a3 = []
  for k in range(10):
    a1.append(numpy.random.normal(size=(3,4)).astype('float32'))
    a2.append(numpy.random.normal(size=(3,4,5)).astype('float64'))
    a3.append((100*numpy.random.normal(size=(2,3,4,5))).astype('int32'))

  arrayset_readwrite('.tensor', a1)
  arrayset_readwrite(".tensor", a2)
  arrayset_readwrite(".tensor", a3)

  # complete transcoding test
  transcode(test_utils.datafile('torch.tensor', __name__))

@test_utils.extension_available('.csv')
def test_csv():

  # array writing tests
  a1 = numpy.random.normal(size=(5,5)).astype('float64')
  a2 = numpy.random.normal(size=(5,10)).astype('float64')
  a3 = numpy.random.normal(size=(5,100)).astype('float64')

  array_readwrite('.csv', a1, close=True)
  array_readwrite(".csv", a2, close=True)
  array_readwrite('.csv', a3, close=True)
  array_readwrite('.csv', a3[::2,::2], close=True) #test non-contiguous

  # arrayset writing tests
  a1 = []
  a2 = []
  a3 = []
  for k in range(10):
    a1.append(numpy.random.normal(size=(5,)).astype('float64'))
    a2.append(numpy.random.normal(size=(50,)).astype('float64'))
    a3.append(numpy.random.normal(size=(500,)).astype('float64'))

  arrayset_readwrite('.csv', a1, close=True)
  arrayset_readwrite(".csv", a2, close=True)
  arrayset_readwrite('.csv', a3, close=True)

