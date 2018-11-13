#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Jun 22 17:50:08 2011 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests for the base HDF5 infrastructure
"""

import os
import sys
import numpy
import random
import nose.tools

from . import HDF5File, load, save, peek_all, test_utils

def read_write_check(outfile, dname, data, dtype=None):
  """Tests scalar input/output on HDF5 files"""

  if dtype is not None: data = [dtype(k) for k in data]

  # First, we test that we can read and write 1 single element
  outfile.append(dname + '_single', data[0])

  # Set attributes on the dataset and current path (single scalar)
  outfile.set_attribute(dname + '_single_attr', data[0], dname + '_single')
  outfile.set_attribute(dname + '_single_attr', data[0])

  # Makes sure we can read the value out
  assert numpy.array_equal(outfile.lread(dname + '_single', 0), data[0])

  # Makes sure we can read the attributes out
  assert numpy.array_equal(outfile.get_attribute(dname + '_single_attr', dname + '_single'), data[0])
  assert numpy.array_equal(outfile.get_attribute(dname + '_single_attr'), data[0])

  # Now we go for the full set
  outfile.append(dname, data)

  # Also create big attributes to see if that works
  outfile.set_attribute(dname + '_attr', data, dname + '_single')
  outfile.set_attribute(dname + '_attr', data)

  # And that we can read it back
  back = outfile.lread(dname) #we read all at once as it is simpler
  for i, b in enumerate(back): assert numpy.array_equal(b, data[i])

  # Check the attributes
  assert numpy.array_equal(outfile.get_attribute(dname + '_attr', dname + '_single'), data)
  assert numpy.array_equal(outfile.get_attribute(dname + '_attr'), data)

def read_write_array_check(outfile, dtype):
  N = 10
  SHAPE = (2, 3, 4, 2) #48 elements in arrays
  arrays = []
  for k in range(N):
    data = [random.uniform(0,N) for z in range(numpy.product(SHAPE))]
    nparray = numpy.array(data, dtype=dtype).reshape(SHAPE)
    arrays.append(nparray)
  read_write_check(outfile, dtype.__name__ + '_array', arrays)

def read_write_subarray_check(outfile, dtype):
  N = 10
  SHAPE = (2, 3, 4, 2) #48 elements in arrays
  arrays = []
  for k in range(N):
    data = [random.uniform(0,N) for z in range(numpy.product(SHAPE))]
    nparray = numpy.array(data, dtype=dtype).reshape(SHAPE)
    usearray = nparray[:,1,1:2,:]
    assert not usearray.flags.contiguous
    arrays.append(usearray)
  read_write_check(outfile, dtype.__name__ + '_subarray', arrays)

def test_can_create():

  # This test demonstrates how to create HDF5 files from scratch,
  # starting from blitz::Arrays

  try:

    # We start by creating some arrays to play with. Please note that in
    # normal cases you are either generating these arrays or reading from
    # other binary files or datasets.
    N = 2
    SHAPE = (3, 2) #6 elements
    NELEMENT = SHAPE[0] * SHAPE[1]
    arrays = []
    for k in range(N):
      data = [int(random.uniform(0,10)) for z in range(NELEMENT)]
      arrays.append(numpy.array(data, 'int32').reshape(SHAPE))

    # Now we create a new binary output file in a temporary location and save
    # the data there.
    tmpname = test_utils.temporary_filename()
    outfile = HDF5File(tmpname, 'w')
    outfile.append('testdata', arrays)

    # Data that is thrown in the file is immediately accessible, so you can
    # interleave read and write operations without any problems.
    # There is a single variable in the file, which is a bob arrayset:
    nose.tools.eq_(outfile.paths(), ['/testdata'])

    # And all the data is *exactly* the same recorded, bit by bit
    back = outfile.lread('testdata') # this is how to read the whole data back
    for i, b in enumerate(back):
      assert numpy.array_equal(b, arrays[i])

    # If you want to immediately close the HDF5 file, just delete the object
    del outfile

    # You can open the file in read-only mode using the 'r' flag. Writing
    # operations on this file will fail.
    readonly = HDF5File(tmpname, 'r')

    # There is a single variable in the file, which is a bob arrayset:
    nose.tools.eq_(readonly.paths(), ['/testdata'])

    # You can get an overview of what is in the HDF5 dataset using the
    # describe() method
    description = readonly.describe('testdata')

    nose.tools.eq_(description[0][0][0], arrays[0].dtype)
    nose.tools.eq_(description[0][0][1], arrays[0].shape)
    nose.tools.eq_(description[0][1], N) #number of elements
    nose.tools.eq_(description[0][2], True) #expandable

    # Test that writing will really fail
    nose.tools.assert_raises(RuntimeError, readonly.append, "testdata", arrays[0])


    # And all the data is *exactly* the same recorded, bit by bit
    back = readonly.lread('testdata') # how to read the whole data back
    for i, b in enumerate(back):
      assert numpy.array_equal(b, arrays[i])

  finally:
    os.unlink(tmpname)

def test_type_support():

  # This test will go through all supported types for reading/writing data
  # from to HDF5 files. One single file will hold all data for this test.
  # This is also supported with HDF5: multiple variables in a single file.

  try:

    N = 100

    tmpname = test_utils.temporary_filename()
    outfile = HDF5File(tmpname, 'w')

    data = [bool(int(random.uniform(0,2))) for z in range(N)]
    read_write_check(outfile, 'bool_data', data)
    data = [int(random.uniform(0,100)) for z in range(N)]
    read_write_check(outfile, 'int_data', data)
    read_write_check(outfile, 'int8_data', data, numpy.int8)
    read_write_check(outfile, 'uint8_data', data, numpy.uint8)
    read_write_check(outfile, 'int16_data', data, numpy.int16)
    read_write_check(outfile, 'uint16_data', data, numpy.uint16)
    read_write_check(outfile, 'int32_data', data, numpy.int32)
    read_write_check(outfile, 'uint32_data', data, numpy.uint32)

    if sys.version_info[0] < 3:
      data = [long(random.uniform(0,1000000000)) for z in range(N)]
    else:
      data = [int(random.uniform(0,1000000000)) for z in range(N)]
    read_write_check(outfile, 'long_data', data)
    read_write_check(outfile, 'int64_data', data, numpy.int64)
    read_write_check(outfile, 'uint64_data', data, numpy.uint64)

    data = [float(random.uniform(0,1)) for z in range(N)]
    read_write_check(outfile, 'float_data', data, float)
    #Note that because of double => float precision issues, the next test will
    #fail. Python floats are actually double precision.
    #read_write_check(outfile, 'float32_data', data, numpy.float32)
    read_write_check(outfile, 'float64_data', data, numpy.float64)
    #The next construction is not supported by bob
    #read_write_check(outfile, 'float128_data', data, numpy.float128)

    data = [complex(random.uniform(0,1),random.uniform(-1,0)) for z in range(N)]
    read_write_check(outfile, 'complex_data', data, complex)
    #Note that because of double => float precision issues, the next test will
    #fail. Python floats are actually double precision.
    #read_write_check(outfile, 'complex64_data', data, numpy.complex64)
    read_write_check(outfile, 'complex128_data', data, numpy.complex128)
    #The next construction is not supported by bob
    #read_write_check(outfile, 'complex256_data', data, numpy.complex256)

    read_write_array_check(outfile, numpy.int8)
    read_write_array_check(outfile, numpy.int16)
    read_write_array_check(outfile, numpy.int32)
    read_write_array_check(outfile, numpy.int64)
    read_write_array_check(outfile, numpy.uint8)
    read_write_array_check(outfile, numpy.uint16)
    read_write_array_check(outfile, numpy.uint32)
    read_write_array_check(outfile, numpy.uint64)
    read_write_array_check(outfile, numpy.float32)
    read_write_array_check(outfile, numpy.float64)
    #read_write_array_check(outfile, numpy.float128) #no numpy conversion
    read_write_array_check(outfile, numpy.complex64)
    read_write_array_check(outfile, numpy.complex128)
    #read_write_array_check(outfile, numpy.complex256) #no numpy conversion

    read_write_subarray_check(outfile, numpy.int8)
    read_write_subarray_check(outfile, numpy.int16)
    read_write_subarray_check(outfile, numpy.int32)
    read_write_subarray_check(outfile, numpy.int64)
    read_write_subarray_check(outfile, numpy.uint8)
    read_write_subarray_check(outfile, numpy.uint16)
    read_write_subarray_check(outfile, numpy.uint32)
    read_write_subarray_check(outfile, numpy.uint64)
    read_write_subarray_check(outfile, numpy.float32)
    read_write_subarray_check(outfile, numpy.float64)
    #read_write_subarray_check(outfile, numpy.float128) #no numpy conversion
    read_write_subarray_check(outfile, numpy.complex64)
    read_write_subarray_check(outfile, numpy.complex128)
    #read_write_subarray_check(outfile, numpy.complex256) #no numpy conversion

  finally:
    os.unlink(tmpname)

def test_dataset_management():

  try:

    # This test examplifies dataset management within HDF5 files and how to
    # copy, delete and move data around.

    # Let's just create some dummy data to play with
    N = 100

    tmpname = test_utils.temporary_filename()
    outfile = HDF5File(tmpname, 'w')

    data = [int(random.uniform(0,N)) for z in range(N)]
    outfile.append('int_data', data)

    # This is how to rename a dataset.
    outfile.rename('int_data', 'MyRenamedDataset')

    # You can move the Dataset to any other hierarchy in the HDF5 file. The
    # directory structure within the file (i.e. the HDF5 groups) will be
    # created on demand.
    outfile.rename('MyRenamedDataset', 'NewDirectory1/Dir2/MyDataset')

    # Let's move the MyDataset dataset to another directory
    outfile.rename('NewDirectory1/Dir2/MyDataset', 'Test2/Bla')

    # So, now the original dataset name does not exist anymore
    nose.tools.eq_(outfile.paths(), ['/Test2/Bla'])

    # We can also unlink the dataset from the file. Please note this will not
    # erase the data in the file, just make it inaccessible
    outfile.unlink('Test2/Bla')

    # Finally, nothing is there anymore
    nose.tools.eq_(outfile.paths(), [])

  finally:
    os.unlink(tmpname)

def test_resize_and_preserve():

  # This test checks that non-contiguous C-style array can be saved
  # into an HDF5 file.

  try:
    # Let's just create some dummy data to play with
    SHAPE = (2, 3) #6 elements
    NELEMENT = SHAPE[0] * SHAPE[1]
    data = [int(random.uniform(0,10)) for z in range(NELEMENT)]
    array = numpy.array(data, 'int32').reshape(SHAPE)

    # Try to save a slice
    tmpname = test_utils.temporary_filename()
    save(array[:,0], tmpname)

  finally:
    os.unlink(tmpname)

def test_can_load_hdf5_from_matlab():

  # shows we can load a 2D matlab array and interpret it as a bunch of 1D
  # arrays, correctly

  t = load(test_utils.datafile('matlab_1d.hdf5', __name__))
  nose.tools.eq_(t.shape, (512,))
  nose.tools.eq_(t.dtype, numpy.float64)

  t = load(test_utils.datafile('matlab_2d.hdf5', __name__))
  nose.tools.eq_(t.shape, (512, 2))
  nose.tools.eq_(t.dtype, numpy.float64)

  # interestingly enough, if you load those files as arrays, you will read
  # the whole data at once:

  dtype, shape, stride = peek_all(test_utils.datafile('matlab_1d.hdf5', __name__))
  nose.tools.eq_(shape, (512,))
  nose.tools.eq_(dtype, numpy.dtype('float64'))

  dtype, shape, stride = peek_all(test_utils.datafile('matlab_2d.hdf5', __name__))
  nose.tools.eq_(shape, (512, 2))
  nose.tools.eq_(dtype, numpy.dtype('float64'))

def test_matlab_import():

  # This test verifies we can import HDF5 datasets generated in Matlab
  mfile = HDF5File(test_utils.datafile('matlab_1d.hdf5', __name__))
  nose.tools.eq_(mfile.paths(), ['/array'])

def test_ioload_unlimited():

  # This test verifies that a 3D array whose first dimension is unlimited
  # and size equal to 1 can be read as a 2D array
  mfile = load(test_utils.datafile('test7_unlimited.hdf5', __name__))
  nose.tools.eq_(mfile.ndim, 2)

def test_attribute_version():

  try:
    tmpname = test_utils.temporary_filename()
    outfile = HDF5File(tmpname, 'w')
    outfile.set_attribute('version', 32)
    nose.tools.eq_(outfile.get_attribute('version'), 32)

  finally:
    os.unlink(tmpname)

def test_string_support():

  try:
    tmpname = test_utils.temporary_filename()
    outfile = HDF5File(tmpname, 'w')
    attribute = 'this is my long test string with \nNew lines'
    outfile.set('string', attribute)
    recovered = outfile.read('string')
    #nose.tools.eq_(attribute, recovered)

  finally:
    del outfile
    os.unlink(tmpname)

def test_string_attribute_support():

  try:
    tmpname = test_utils.temporary_filename()
    outfile = HDF5File(tmpname, 'w')
    attribute = 'this is my long test string with \nNew lines'
    outfile.set_attribute('string', attribute)
    recovered = outfile.get_attribute('string')
    nose.tools.eq_(attribute, recovered)

    data = [1,2,3,4,5]
    outfile.set('data', data)
    outfile.set_attribute('string', attribute, 'data')
    recovered = outfile.get_attribute('string', 'data')
    nose.tools.eq_(attribute, recovered)

  finally:
    os.unlink(tmpname)

def test_can_use_set_with_iterables():

  try:
    tmpname = test_utils.temporary_filename()
    outfile = HDF5File(tmpname, 'w')
    data = [1, 34.5, True]
    outfile.set('data', data)
    assert numpy.array_equal(data, outfile.read('data'))

  finally:
    os.unlink(tmpname)

def test_has_attribute():

  try:
    tmpname = test_utils.temporary_filename()
    outfile = HDF5File(tmpname, 'w')
    i = 35
    f = 3.14
    outfile.set_attribute('int', i)
    outfile.set_attribute('float', f)
    assert outfile.has_attribute('int')
    nose.tools.eq_(outfile.get_attribute('int'), 35)
    assert outfile.has_attribute('float')
    nose.tools.eq_(outfile.get_attribute('float'), 3.14)

  finally:
    os.unlink(tmpname)

def test_get_attributes():

  try:
    tmpname = test_utils.temporary_filename()
    outfile = HDF5File(tmpname, 'w')
    nothing = outfile.get_attributes()
    nose.tools.eq_(len(nothing), 0)
    assert isinstance(nothing, dict)
    i = 35
    f = 3.14
    outfile.set_attribute('int', i)
    outfile.set_attribute('float', f)
    d = outfile.get_attributes()
    nose.tools.eq_(d['int'], i)
    nose.tools.eq_(d['float'], f)

  finally:
    os.unlink(tmpname)

def test_set_compression():

  try:

    tmpname = test_utils.temporary_filename()
    outfile = HDF5File(tmpname, 'w')
    data = numpy.random.random((50,50))
    outfile.set('data', data, compression=9)
    recovered = outfile.read('data')
    assert numpy.array_equal(data, recovered)
    del outfile

  finally:

    os.unlink(tmpname)

def test_append_compression():

  try:

    tmpname = test_utils.temporary_filename()
    outfile = HDF5File(tmpname, 'w')
    data = numpy.random.random((50,50))
    for k in range(len(data)): outfile.append('data', data[k], compression=9)
    recovered = outfile.read('data')
    assert numpy.array_equal(data, recovered)

  finally:

    os.unlink(tmpname)


def test_close():
  try:
    tmpname = test_utils.temporary_filename()
    outfile = HDF5File(tmpname, 'w')
    outfile.close()

    def test_set():
      outfile.set("Test", numpy.array([1,2]))
    nose.tools.assert_raises(RuntimeError, test_set)

    def test_read():
      test = outfile.read("Test")
    nose.tools.assert_raises(RuntimeError, test_read)

    def test_filename():
      fn = outfile.filename
    nose.tools.assert_raises(RuntimeError, test_filename)

  finally:
    os.unlink(tmpname)

def test_copy_constructor():
  try:
    tmpname = test_utils.temporary_filename()
    tmpname2 = test_utils.temporary_filename()
    hdf5 = HDF5File(tmpname, 'w')
    shallow = HDF5File(hdf5)
    deep = HDF5File(tmpname2, 'w')
    hdf5.copy(deep)

    hdf5.create_group("Test")
    hdf5.cd("Test")
    hdf5.set("Data", numpy.random.random((10,10)))
    hdf5.flush()

    assert shallow.has_group("/Test")
    assert shallow.has_key("/Test/Data")
    assert hdf5.filename == shallow.filename
    assert hdf5.keys() == shallow.keys()
    assert hdf5.cwd == shallow.cwd

    assert not deep.has_group("/Test")
    assert hdf5.filename != deep.filename
    assert hdf5.keys() != deep.keys()
    assert hdf5.cwd != deep.cwd
    assert str(shallow)

    hdf5.cd("..")

    assert hdf5.sub_groups() == shallow.sub_groups()
    assert hdf5.sub_groups() != deep.sub_groups()

    assert hdf5.writable
    assert shallow.writable
    assert deep.writable


    hdf5.close()
    deep.close()

    def test_filename():
      fn = shallow.filename
    nose.tools.assert_raises(RuntimeError, test_filename)
    def test_repr():
      fn = str(shallow)
    nose.tools.assert_raises(RuntimeError, test_repr)

  finally:
    os.unlink(tmpname)
    os.unlink(tmpname2)


def test_python_interfaces():
  try:
    tmpname = test_utils.temporary_filename()
    with HDF5File(tmpname, 'w') as hdf5:
      a = numpy.arange(10)
      b = a * 2
      hdf5['Data'] = a
      hdf5.create_group('Group1')
      hdf5['Group1/Data'] = b
      assert "/Group1/Data" in hdf5
      assert [key for key in hdf5] == hdf5.keys()
      assert numpy.allclose(hdf5['Data'], hdf5.get('Data'))
      assert all(numpy.allclose(c, d) for c, d in zip(hdf5.values(), (a, b)))
      for key, value in hdf5.items():
        assert numpy.allclose(value, hdf5[key])

  finally:
    os.unlink(tmpname)

def unicode_test():
  filename = u"Φîłèñäϻæ.hdf5"

  # writing using unicode filename
  hdf5 = HDF5File(filename, 'w')
  hdf5.set('Test', range(10))
  del hdf5

  # writing using unicode filename
  hdf5 = HDF5File(filename)
  assert numpy.allclose(hdf5["Test"], range(10))
  os.remove(filename)

