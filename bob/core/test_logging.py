#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 09 Jul 2013 13:24:49 CEST

"""Tests for the logging subsystem
"""

import bob.extension.log
import bob.core

def test_from_python():
  logger = bob.core.log.setup("bob.core")
  bob.core.log.set_verbosity_level(logger, 2)

  # send a log message
  logger.debug("This is a test debug message")
  logger.info("This is a test info message")
  bob.core.log.set_verbosity_level(logger, 0)
  logger.warn("This is a test warn message")
  logger.error("This is a test error message")


def test_from_python_output():
  import logging, sys

  if sys.version_info[0] < 3:
    from StringIO import StringIO
  else:
    from io import StringIO
  # create an artificial logger using the logging module
  logger = logging.getLogger("XXX.YYY")

  # add handlers
  out, err = StringIO(), StringIO()
  _warn_err = logging.StreamHandler(err)
  _warn_err.setLevel(logging.WARNING)
  logger.addHandler(_warn_err)

  _debug_info = logging.StreamHandler(out)
  _debug_info.setLevel(logging.DEBUG)
  _debug_info.addFilter(bob.extension.log._InfoFilter())
  logger.addHandler(_debug_info)

  # now, set up the logger
  logger = bob.core.log.setup("XXX.YYY")

  # send log messages
  bob.core.log.set_verbosity_level(logger, 2)
  logger.debug("This is a test debug message")
  logger.info("This is a test info message")

  bob.core.log.set_verbosity_level(logger, 0)
  logger.warn("This is a test warn message")
  logger.error("This is a test error message")

  out = out.getvalue().rstrip()
  err = err.getvalue().rstrip()

  assert out.startswith("XXX.YYY")
  assert "INFO" in out
  assert out.endswith("This is a test info message")

  assert err.startswith("XXX.YYY")
  assert "ERROR" in err
  assert err.endswith("This is a test error message")


def test_from_cxx():
  from bob.core._test import _test_log_message
  _test_log_message(1, 'error', 'this is a test message')

def test_from_cxx_multithreaded():
  from bob.core._test import _test_log_message_mt
  _test_log_message_mt(2, 1, 'error', 'this is a test message')



def test_from_cxx_disable():
  from bob.core._test import _test_output_disable
  _test_output_disable()
