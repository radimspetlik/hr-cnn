from functools import partial
import numpy as np
import tempfile
from bob.extension.processors import (
    SequentialProcessor, ParallelProcessor)

DATA = [0, 1, 2, 3, 4]
PROCESSORS = [partial(np.power, 2), np.mean]
SEQ_DATA = PROCESSORS[1](PROCESSORS[0](DATA))
PAR_DATA = (PROCESSORS[0](DATA), PROCESSORS[1](DATA))


def test_processors():
  proc = SequentialProcessor(PROCESSORS)
  data = proc(DATA)
  assert np.allclose(data, SEQ_DATA)

  proc = ParallelProcessor(PROCESSORS)
  data = proc(DATA)
  assert all(np.allclose(x1, x2) for x1, x2 in zip(data, PAR_DATA))
