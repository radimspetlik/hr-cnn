#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import os
import sys
import shutil
import tempfile

if sys.version_info[0] <= 2:
  import urllib2 as urllib
else:
  import urllib.request as urllib

import nose.tools

from ..driver import download


if 'DOCSERVER' in os.environ:
  USE_SERVER=os.environ['DOCSERVER']
else:
  USE_SERVER='https://www.idiap.ch'


class Namespace(object):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


def test_download_banca():
  tmpdir = tempfile.mkdtemp()
  try:
    arguments = Namespace(
        files=[tmpdir+'/db.sql3'],
        force=False,
        missing=False,
        name='banca',
        test_dir=tmpdir,
        version='0.9.0',
        source='%s/software/bob/databases/latest/' % USE_SERVER,
        )
    download(arguments)
  finally:
    shutil.rmtree(tmpdir)


@nose.tools.raises(urllib.HTTPError)
def test_download_unexisting():
  tmpdir = tempfile.mkdtemp()
  try:
    arguments = Namespace(
        files=[tmpdir+'/db.sql3'],
        force=False,
        missing=False,
        name='does_not_exist',
        test_dir=tmpdir,
        version='0.9.0',
        source='%s/software/bob/databases/latest/' % USE_SERVER,
        )
    download(arguments)
  finally:
    shutil.rmtree(tmpdir)


def test_download_not_missing():
  tmpdir = tempfile.mkdtemp()
  try:
    arguments = Namespace(
        files=[tmpdir+'/db.sql3'],
        force=False,
        missing=False,
        name='banca',
        test_dir=tmpdir,
        version='0.9.0',
        source='%s/software/bob/databases/latest/' % USE_SERVER,
        )
    download(arguments)
    arguments = Namespace(
        files=[tmpdir+'/db.sql3'],
        force=False,
        missing=True,
        name='banca',
        test_dir=tmpdir,
        version='0.9.0',
        source='%s/software/bob/databases/latest/' % USE_SERVER,
        )
    download(arguments)
  finally:
    shutil.rmtree(tmpdir)


def test_download_missing():
  tmpdir = tempfile.mkdtemp()
  try:
    arguments = Namespace(
        files=[tmpdir+'/db.sql3'],
        force=False,
        missing=True,
        name='banca',
        test_dir=tmpdir,
        version='0.9.0',
        source='%s/software/bob/databases/latest/' % USE_SERVER,
        )
    download(arguments)
  finally:
    shutil.rmtree(tmpdir)


def test_download_force():
  tmpdir = tempfile.mkdtemp()
  try:
    arguments = Namespace(
        files=[tmpdir+'/db.sql3'],
        force=False,
        missing=False,
        name='banca',
        test_dir=tmpdir,
        version='0.9.0',
        source='%s/software/bob/databases/latest/' % USE_SERVER,
        )
    download(arguments)
    arguments = Namespace(
        files=[tmpdir+'/db.sql3'],
        force=True,
        missing=False,
        name='banca',
        test_dir=tmpdir,
        version='0.9.0',
        source='%s/software/bob/databases/latest/' % USE_SERVER,
        )
    download(arguments)
  finally:
    shutil.rmtree(tmpdir)


@nose.tools.raises(IOError)
def test_download_existing():
  tmpdir = tempfile.mkdtemp()
  try:
    arguments = Namespace(
        files=[tmpdir+'/db.sql3'],
        force=False,
        missing=False,
        name='banca',
        test_dir=tmpdir,
        version='0.9.0',
        source='%s/software/bob/databases/latest/' % USE_SERVER,
        )
    download(arguments)
    arguments = Namespace(
        files=[tmpdir+'/db.sql3'],
        force=False,
        missing=False,
        name='banca',
        test_dir=tmpdir,
        version='0.9.0',
        source='%s/software/bob/databases/latest/' % USE_SERVER,
        )
    download(arguments)
  finally:
    shutil.rmtree(tmpdir)
