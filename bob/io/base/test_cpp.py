from bob.io.base._test import _test_api

import tempfile
import shutil

def test_api():
  temp_dir = tempfile.mkdtemp()
  try:
    _test_api(temp_dir)
  finally:
    shutil.rmtree(temp_dir)
