#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 15 May 13:31:35 2014 CEST
#
# Copyright (C) Idiap Research Institute, Martigny, Switzerland

"""Re-usable decorators and utilities for bob test code
"""

import functools
import nose.plugins.skip
from distutils.version import StrictVersion as SV

from bob.io.base.test_utils import datafile, temporary_filename

# Here is a table of ffmpeg versions against libavcodec, libavformat and
# libavutil versions
ffmpeg_versions = {
    '0.5':  [ SV('52.20.0'),   SV('52.31.0'),   SV('49.15.0')   ],
    '0.6':  [ SV('52.72.2'),   SV('52.64.2'),   SV('50.15.1')   ],
    '0.7':  [ SV('52.122.0'),  SV('52.110.0'),  SV('50.43.0')   ],
    '0.8':  [ SV('53.7.0'),    SV('53.4.0'),    SV('51.9.1')    ],
    '0.9':  [ SV('53.42.0'),   SV('53.24.0'),   SV('51.32.0')   ],
    '0.10': [ SV('53.60.100'), SV('53.31.100'), SV('51.34.101') ],
    '0.11': [ SV('54.23.100'), SV('54.6.100'),  SV('51.54.100') ],
    '1.0':  [ SV('54.59.100'), SV('54.29.104'), SV('51.73.101') ],
    '1.1':  [ SV('54.86.100'), SV('54.59.106'), SV('52.13.100') ],
    '1.2':  [ SV('54.92.100'), SV('54.63.104'), SV('52.18.100') ],
    '2.0':  [ SV('55.18.102'), SV('55.12.100'), SV('52.38.100') ],
    '2.1':  [ SV('55.39.100'), SV('55.19.104'), SV('52.48.100') ],
    }

def ffmpeg_version_lessthan(v):
  '''Returns true if the version of ffmpeg compiled-in is at least the version
  indicated as a string parameter.'''

  from .version import externals
  avcodec_inst= SV(externals['FFmpeg']['avcodec'])
  avcodec_req = ffmpeg_versions[v][0]
  return avcodec_inst < avcodec_req

def codec_available(codec):
  '''Decorator to check if a codec is available before enabling a test'''

  def test_wrapper(test):

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
      from . import supported_video_codecs
      d = supported_video_codecs()
      if codec in d and d[codec]['encode'] and d[codec]['decode']:
        return test(*args, **kwargs)
      else:
        raise nose.plugins.skip.SkipTest('A functional codec for "%s" is not installed with FFmpeg' % codec)

    return wrapper

  return test_wrapper
