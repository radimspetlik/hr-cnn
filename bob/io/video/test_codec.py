#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 12 Nov 09:25:56 2013

"""Runs quality tests on video codecs supported
"""

import os
import numpy
import nose.tools
from . import test_utils
from .utils import color_distortion, frameskip_detection, quality_degradation

# These are some global parameters for the test.
INPUT_VIDEO = test_utils.datafile('test.mov', __name__)

# Fix numpy random seed for tests, so they are repeatable
numpy.random.seed(0)

def check_format_codec(function, shape, framerate, format, codec, maxdist):

  length, height, width = shape
  fname = test_utils.temporary_filename(suffix='.%s' % format)

  try:
    orig, framerate, encoded = function(shape, framerate, format, codec, fname)
    reloaded = encoded.load()

    # test number of frames is correct
    assert len(orig) == len(encoded), "original length %d != %d encoded for format `%s' and codec `%s'" % (len(orig), len(encoded), format, codec)
    assert len(orig) == len(reloaded), "original length %d != %d reloaded for format `%s' and codec `%s'" % (len(orig), len(reloaded), format, codec)

    # test distortion patterns (quick sequential check)
    dist = []
    for k, of in enumerate(orig):
      dist.append(abs(of.astype('float64')-reloaded[k].astype('float64')).mean())
    assert max(dist) <= maxdist, "max(distortion) %g > %g allowed for format `%s' and codec `%s'" % (max(dist), maxdist, format, codec)

    # assert we can randomly access any frame (choose 3 at random)
    for k in numpy.random.randint(length, size=(3,)):
      rdist = abs(orig[k].astype('float64')-encoded[k].astype('float64')).mean()
      assert rdist <= maxdist, "distortion(frame[%d]) %g > %g allowed for format `%s' and codec `%s'" % (k, rdist, maxdist, format, codec)

    # make sure that the encoded frame rate is not off by a big amount
    assert abs(framerate - encoded.frame_rate) <= (1.0/length), "reloaded framerate %g differs from original %g by more than %g for format `%s' and codec `%s'" % (encoded.frame_rate, framerate, 1.0/length, format, codec)

  finally:

    if os.path.exists(fname): os.unlink(fname)

def test_format_codecs():

  length = 30
  width = 128
  height = 128
  framerate = 30.
  shape = (length, height, width)
  methods = dict(
      frameskip = frameskip_detection,
      color     = color_distortion,
      noise     = quality_degradation,
      )

  # distortion patterns for specific codecs
  distortions = dict(
      # we require high standards by default
      default    = dict(frameskip=0.1,  color=9.0,  noise=45.),

      # high-quality encoders
      zlib        = dict(frameskip=0.0,  color=0.0, noise=0.0),
      ffv1        = dict(frameskip=0.05, color=9.0,  noise=46.),
      vp8         = dict(frameskip=0.3,  color=9.0, noise=65.),
      libvpx      = dict(frameskip=0.3,  color=9.0, noise=65.),
      h264        = dict(frameskip=0.5,  color=9.0, noise=55.),
      libx264     = dict(frameskip=0.4,  color=9.0, noise=50.),
      libopenh264 = dict(frameskip=0.5,  color=9.0, noise=55.),
      theora      = dict(frameskip=0.5,  color=9.0, noise=70.),
      libtheora   = dict(frameskip=0.5,  color=9.0, noise=70.),
      mpeg4       = dict(frameskip=1.0,  color=9.0, noise=55.),

      # older, but still good quality encoders
      mjpeg      = dict(frameskip=1.2,  color=9.0, noise=50.),
      mpegvideo  = dict(frameskip=1.3,  color=9.0, noise=80.),
      mpeg2video = dict(frameskip=1.3,  color=9.0, noise=80.),
      mpeg1video = dict(frameskip=1.4,  color=9.0, noise=50.),

      # low quality encoders - avoid using - available for compatibility
      wmv2       = dict(frameskip=3.0,  color=10., noise=50.),
      wmv1       = dict(frameskip=2.5,  color=10., noise=50.),
      msmpeg4    = dict(frameskip=6.,   color=10., noise=50.),
      msmpeg4v2  = dict(frameskip=6.,   color=10., noise=50.),
      )

  from . import supported_videowriter_formats
  SUPPORTED = supported_videowriter_formats()
  for format in SUPPORTED:
    for codec in SUPPORTED[format]['supported_codecs']:
      for method in methods:
        check_format_codec.description = "%s.test_%s_format_%s_codec_%s" % (__name__, method, format, codec)
        distortion = distortions.get(codec, distortions['default'])[method]
        yield check_format_codec, methods[method], shape, framerate, format, codec, distortion

def check_user_video(format, codec, maxdist):

  from . import reader, writer
  fname = test_utils.temporary_filename(suffix='.%s' % format)
  MAXLENTH = 10 #use only the first 10 frames

  try:

    orig_vreader = reader(INPUT_VIDEO)
    orig = orig_vreader[:MAXLENTH]
    (olength, _, oheight, owidth) = orig.shape
    assert len(orig) == MAXLENTH, "original length %d != %d MAXLENTH for format `%s' and codec `%s'" % (len(orig), MAXLENTH, format, codec)

    # encode the input video using the format and codec provided by the user
    outv = writer(fname, oheight, owidth, orig_vreader.frame_rate,
        codec=codec, format=format, check=True)
    for k in orig: outv.append(k)
    outv.close()

    # reload from saved file
    encoded = reader(fname)
    reloaded = encoded.load()

    # test number of frames is correct
    assert len(orig) == len(encoded), "original length %d != %d encoded for format `%s' and codec `%s'" % (len(orig), len(encoded), format, codec)
    assert len(orig) == len(reloaded), "original length %d != %d reloaded for format `%s' and codec `%s'" % (len(orig), len(reloaded), format, codec)

    # test distortion patterns (quick sequential check)
    dist = []
    for k, of in enumerate(orig):
      dist.append(abs(of.astype('float64')-reloaded[k].astype('float64')).mean())
    assert max(dist) <= maxdist, "max(distortion) %g > %g allowed for format `%s' and codec `%s'" % (max(dist), maxdist, format, codec)

    # make sure that the encoded frame rate is not off by a big amount
    assert abs(orig_vreader.frame_rate - encoded.frame_rate) <= (1.0/MAXLENTH), "original video framerate %g differs from encoded %g by more than %g for format `%s' and codec `%s'" % (encoded.frame_rate, orig_vreader.framerate, 1.0/MAXLENTH, format, codec)

  finally:

    if os.path.exists(fname): os.unlink(fname)

def test_user_video():

  # distortion patterns for specific codecs
  distortions = dict(
      # we require high standards by default
      default    = 1.5,

      # high-quality encoders
      zlib        = 0.0,
      ffv1        = 1.7,
      vp8         = 2.7,
      libvpx      = 2.7,
      h264        = 2.7,
      libx264     = 2.5,
      libopenh264 = 3.0,
      theora      = 2.0,
      libtheora   = 2.0,
      mpeg4       = 2.3,

      # older, but still good quality encoders
      mjpeg      = 1.8,
      mpegvideo  = 2.3,
      mpeg2video = 2.3,
      mpeg1video = 2.3,

      # low quality encoders - avoid using - available for compatibility
      wmv2       = 2.3,
      wmv1       = 2.3,
      msmpeg4    = 2.3,
      msmpeg4v2  = 2.3,
      )

  from . import supported_videowriter_formats
  SUPPORTED = supported_videowriter_formats()
  for format in SUPPORTED:
    for codec in SUPPORTED[format]['supported_codecs']:
      check_user_video.description = "%s.test_user_video_format_%s_codec_%s" % (__name__, format, codec)
      distortion = distortions.get(codec, distortions['default'])
      yield check_user_video, format, codec, distortion
