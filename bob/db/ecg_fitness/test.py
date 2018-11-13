#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri Mar 14 14:44:31 CET 2014

"""A few checks at the COHFACE dataset
"""

import os, sys
import unittest
import nose.tools
import pkg_resources

from . import Database
from .driver import DATABASE_LOCATION


def db_available(test):
  """Decorator for detecting if we're running the test at Idiap"""
  from nose.plugins.skip import SkipTest
  import functools

  @functools.wraps(test)
  def wrapper(*args, **kwargs):

    if os.path.exists(DATABASE_LOCATION):
      return test(*args, **kwargs)
    else:
      raise SkipTest("Raw database files are not available")

  return wrapper


def meta_available(test):
  """Decorator for detecting if we're running the test on an annotated db"""
  from nose.plugins.skip import SkipTest
  import functools

  @functools.wraps(test)
  def wrapper(*args, **kwargs):

    if os.path.exists(pkg_resources.resource_filename(__name__, 'data')):
      return test(*args, **kwargs)
    else:
      raise SkipTest("Annotation files are not available")

  return wrapper


class CohfaceTest(unittest.TestCase):
  """Performs various tests on the COHFACE database."""

  def setUp(self):
    self.db = Database()


  @db_available
  def test01_objects(self):
    self.assertEqual(len(self.db.objects()), 160)


  @db_available
  def test02_can_read_hdf5(self):

    for obj in self.db.objects()[:3]:

      path = obj.make_path(DATABASE_LOCATION, '.hdf5')
      self.assertTrue(os.path.exists(path))

      f = obj.load_hdf5(DATABASE_LOCATION)
      attr = obj.metadata(DATABASE_LOCATION)

      assert attr

      '''
      time = len(signal)/freq

      # correlation between video data and physiological signal
      if abs(time-obj.duration) > 2:
        print('Physiological signal (%d seconds) is very different in size from estimated video duration (%d seconds) on sample `%s/%s\'' % (time, obj.duration, obj.basedir, obj.stem))
      '''


  @db_available
  def test03_can_read_video(self):

    for obj in self.db.objects()[:5]:

      video = obj.load_video(DATABASE_LOCATION)
      assert video.number_of_frames


  @db_available
  def test04_can_estimate_heart_rate(self):

    for obj in self.db.objects()[:10]:

      hr = obj.estimate_heartrate_in_bpm(DATABASE_LOCATION)


  @meta_available
  def test05_can_read_meta(self):

    for obj in self.db.objects()[:3]:

      detection = obj.load_face_detection()
      assert detection

      hr = obj.load_heart_rate_in_bpm()
      assert hr


  @nose.tools.nottest
  @db_available
  def test06_can_plot(self):
    import matplotlib.pyplot as plt
    from .utils import plot_signal_rr, plot_signal_hr

    for obj in self.db.objects()[:3]:
      f = obj.load_hdf5(DATABASE_LOCATION)

      plt.subplot(2,1,1)
      avg_hr, peaks = plot_signal_hr(f.get('pulse'),
          float(f.get_attribute('sample-rate-hz')))

      plt.subplot(2,1,2)
      avg_rr, peaks = plot_signal_rr(f.get('respiration'),
          float(f.get_attribute('sample-rate-hz')))

      plt.tight_layout()
      plt.show()


class CmdLineTest(unittest.TestCase):
  """Makes sure our command-line is working properly."""

  def test01_manage_dumplist(self):

    from bob.db.base.script.dbmanage import main

    args = [
            'cohface',
            'dumplist',
            '--self-test',
            ]

    self.assertEqual(main(args), 0)


  @db_available
  def test02_manage_checkfiles(self):

    from bob.db.base.script.dbmanage import main

    args = [
            'cohface',
            'checkfiles',
            '--self-test',
            '--directory=%s' % DATABASE_LOCATION,
            ]

    self.assertEqual(main(args), 0)


  @db_available
  def notest03_can_create_meta(self):

    from bob.db.base.script.dbmanage import main

    args = [
            'cohface',
            'mkmeta',
            '--self-test',
            '--directory=%s' % DATABASE_LOCATION,
            ]

    self.assertEqual(main(args), 0)
