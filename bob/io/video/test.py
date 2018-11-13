#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Jun 22 17:50:08 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Runs some video tests
"""

import os
import numpy
import nose.tools
from . import test_utils

from bob.io.base import load

# These are some global parameters for the test.
INPUT_VIDEO = test_utils.datafile('test.mov', __name__)


def test_codec_support():

  # Describes all encoders
  from . import describe_encoder, describe_decoder, supported_video_codecs

  supported = supported_video_codecs()

  for k,v in supported.items():
    # note: searching by name (using `k') will not always work
    if v['decode']: assert describe_decoder(v['id'])
    if v['encode']: assert describe_encoder(v['id'])

  # Assert we support, at least, some known codecs
  for codec in ('ffv1', 'wmv2', 'mpeg4', 'mjpeg', 'h264'):
    assert codec in supported
    assert supported[codec]['encode']
    assert supported[codec]['decode']


def test_input_format_support():

  # Describes all encoders
  from . import supported_videoreader_formats

  supported = supported_videoreader_formats()

  # Assert we support, at least, some known codecs
  for fmt in ('avi', 'mov', 'mp4'):
    assert fmt in supported


def test_output_format_support():

  # Describes all encoders
  from . import supported_videowriter_formats

  supported = supported_videowriter_formats()

  # Assert we support, at least, some known codecs
  for fmt in ('avi', 'mov', 'mp4'):
    assert fmt in supported


def test_video_reader_attributes():

  from . import reader

  iv = reader(INPUT_VIDEO)

  assert isinstance(iv.filename, str)
  assert isinstance(iv.height, int)
  assert isinstance(iv.width, int)
  assert iv.height != iv.width
  assert isinstance(iv.duration, int)
  assert isinstance(iv.format_name, str)
  assert isinstance(iv.format_long_name, str)
  assert isinstance(iv.codec_name, str)
  assert isinstance(iv.codec_long_name, str)
  assert isinstance(iv.frame_rate, float)
  assert isinstance(iv.video_type, tuple)
  assert len(iv.video_type) == 3
  assert isinstance(iv.video_type[0], numpy.dtype)
  assert isinstance(iv.video_type[1], tuple)
  assert isinstance(iv.video_type[2], tuple)
  assert isinstance(iv.frame_type, tuple)
  assert len(iv.frame_type) == 3
  assert iv.frame_type[0] == iv.video_type[0]
  assert isinstance(iv.video_type[1], tuple)
  nose.tools.eq_(len(iv.video_type[1]), len(iv.frame_type[1])+1)
  nose.tools.eq_(len(iv.video_type[2]), len(iv.frame_type[2])+1)
  assert isinstance(iv.info, str)


def write_unicode_temp_file():

  prefix = 'bobtest_straße_'
  suffix = '.avi'

  tmpname = test_utils.temporary_filename(prefix=prefix, suffix=suffix)

  # Writing temp file for testing
  from . import writer

  width = 20
  height = 20
  framerate = 24
  outv = writer(tmpname, height, width, framerate)

  for i in range(0, 3):
    newframe = (numpy.random.random_integers(0,255,(3,height,width)))
    outv.append(newframe.astype('uint8'))

  outv.close()
  return tmpname


def test_video_reader_unicode():

  try:

    # Writing temp file for testing
    tmpname = write_unicode_temp_file()

    from . import reader

    iv = reader(tmpname)

    assert isinstance(iv.filename, str)
    assert 'ß' in tmpname
    assert 'ß' in iv.filename

  finally:
    if os.path.exists(tmpname): os.unlink(tmpname)


def test_video_reader_str():

  from . import reader

  iv = reader(INPUT_VIDEO)
  assert repr(iv)
  assert str(iv)


def test_can_iterate():

  from . import reader
  video = reader(INPUT_VIDEO)
  counter = 0
  for frame in video:
    assert isinstance(frame, numpy.ndarray)
    assert len(frame.shape) == 3
    assert frame.shape[0] == 3 #color-bands (RGB)
    assert frame.shape[1] == 240 #height
    assert frame.shape[2] == 320 #width
    counter += 1

  assert counter == len(video) #we have gone through all frames


def test_iteration():

  from . import reader
  f = reader(INPUT_VIDEO)
  objs = load(INPUT_VIDEO)

  nose.tools.eq_(len(f), len(objs))
  for l, i in zip(objs, f):
    assert numpy.allclose(l, i)


def test_base_load_on_unicode():

  try:

    # Writing temp file for testing
    tmpname = write_unicode_temp_file()

    from . import reader
    f = reader(tmpname)
    objs = load(tmpname)

    nose.tools.eq_(len(f), len(objs))
    for l, i in zip(objs, f):
      assert numpy.allclose(l.shape, i.shape)

  finally:
    if os.path.exists(tmpname): os.unlink(tmpname)


def test_indexing():

  from . import reader
  f = reader(INPUT_VIDEO)

  nose.tools.eq_(len(f), 375)

  objs = f[:10]
  nose.tools.eq_(len(objs), 10)
  obj0 = f[0]
  obj1 = f[1]

  # simple indexing
  assert numpy.allclose(objs[0], obj0)
  assert numpy.allclose(objs[1], obj1)
  assert numpy.allclose(f[len(f)-1], f[-1])
  assert numpy.allclose(f[len(f)-2], f[-2])


def test_slicing_empty():

  from . import reader
  f = reader(INPUT_VIDEO)

  objs = f[1:1]
  assert objs.shape == tuple()
  assert objs.dtype == numpy.uint8


def test_slicing_0():

  from . import reader
  f = reader(INPUT_VIDEO)

  objs = f[:]
  for i, k in enumerate(load(INPUT_VIDEO)):
    assert numpy.allclose(k, objs[i])


def test_slicing_1():

  from . import reader
  f = reader(INPUT_VIDEO)

  s = f[3:10:2]
  nose.tools.eq_(len(s), 4)
  assert numpy.allclose(s[0], f[3])
  assert numpy.allclose(s[1], f[5])
  assert numpy.allclose(s[2], f[7])
  assert numpy.allclose(s[3], f[9])


def test_slicing_2():

  from . import reader
  f = reader(INPUT_VIDEO)

  s = f[-10:-2:3]
  nose.tools.eq_(len(s), 3)
  assert numpy.allclose(s[0], f[len(f)-10])
  assert numpy.allclose(s[1], f[len(f)-7])
  assert numpy.allclose(s[2], f[len(f)-4])


def test_slicing_3():

  from . import reader
  f = reader(INPUT_VIDEO)
  objs = f.load()

  # get negative stepping slice
  s = f[20:10:-3]
  nose.tools.eq_(len(s), 4)
  assert numpy.allclose(s[0], f[20])
  assert numpy.allclose(s[1], f[17])
  assert numpy.allclose(s[2], f[14])
  assert numpy.allclose(s[3], f[11])


def test_slicing_4():

  from . import reader
  f = reader(INPUT_VIDEO)
  objs = f[:21]

  # get all negative slice
  s = f[-10:-20:-3]
  nose.tools.eq_(len(s), 4)
  assert numpy.allclose(s[0], f[len(f)-10])
  assert numpy.allclose(s[1], f[len(f)-13])
  assert numpy.allclose(s[2], f[len(f)-16])
  assert numpy.allclose(s[3], f[len(f)-19])


def test_can_use_array_interface():

  from . import reader
  array = load(INPUT_VIDEO)
  iv = reader(INPUT_VIDEO)

  for frame_id, frame in zip(range(array.shape[0]), iv.__iter__()):
    assert numpy.array_equal(array[frame_id,:,:,:], frame)


def test_video_reading_after_writing():

  from . import test_utils
  tmpname = test_utils.temporary_filename(suffix='.avi')

  from . import writer, reader

  try:

    width = 20
    height = 20
    framerate = 24

    outv = writer(tmpname, height, width, framerate)
    for i in range(0, 3):
      newframe = (numpy.random.random_integers(0,255,(3,height,width)))
      outv.append(newframe.astype('uint8'))
    outv.close()

    # this should not crash
    i = reader(tmpname)
    nose.tools.eq_(i.number_of_frames, 3)
    nose.tools.eq_(i.width, width)
    nose.tools.eq_(i.height, height)

  finally:
    # And we erase both files after this
    if os.path.exists(tmpname): os.unlink(tmpname)


def test_video_writer_close():

  from . import test_utils
  tmpname = test_utils.temporary_filename(suffix='.avi')

  from . import writer, reader

  try:

    width = 20
    height = 20
    framerate = 24

    outv = writer(tmpname, height, width, framerate)
    for i in range(0, 3):
      newframe = (numpy.random.random_integers(0,255,(3,height,width)))
      outv.append(newframe.astype('uint8'))
    outv.close()

    # this should not crash
    nose.tools.eq_(outv.filename, tmpname)
    nose.tools.eq_(outv.width, width)
    nose.tools.eq_(outv.height, height)
    nose.tools.eq_(len(outv), 3)
    nose.tools.eq_(outv.number_of_frames, len(outv))
    nose.tools.eq_(outv.frame_rate, framerate)
    assert outv.bit_rate
    assert outv.gop

  finally:
    # And we erase both files after this
    if os.path.exists(tmpname): os.unlink(tmpname)


def test_closed_video_writer_raises():

  from . import test_utils
  tmpname = test_utils.temporary_filename(suffix='.avi')

  from . import writer

  try:

    width = 20
    height = 20
    framerate = 24

    outv = writer(tmpname, height, width, framerate)
    for i in range(0, 3):
      newframe = (numpy.random.random_integers(0,255,(3,height,width)))
      outv.append(newframe.astype('uint8'))
    outv.close()

    nose.tools.assert_raises(RuntimeError, outv.__str__)
    nose.tools.assert_raises(RuntimeError, outv.__repr__)
    nose.tools.assert_raises(RuntimeError, outv.append, newframe)

    del outv

  finally:
    # And we erase both files after this
    if os.path.exists(tmpname): os.unlink(tmpname)
