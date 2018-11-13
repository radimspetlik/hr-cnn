#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import os
import logging
logger = logging.getLogger(__name__)

_idiap_annotations = {
    1: 'reyeo',
    2: 'reyet',
    3: 'reyep',
    4: 'reyeb',
    5: 'reyei',
    6: 'leyei',
    7: 'leyet',
    8: 'leyep',
    9: 'leyeb',
    10: 'leyeo',
    11: 'rbrowo',
    12: 'rbrowi',
    13: 'lbrowi',
    14: 'lbrowo',
    15: 'noser',
    16: 'noset',
    17: 'nosel',
    18: 'mouthr',
    19: 'moutht',
    20: 'mouthb',
    21: 'mouthl',
    22: 'chin'
}


def read_annotation_file(file_name, annotation_type):
  """This function provides default functionality to read annotation files.


  Parameters
  ----------
  file_name : str
      The full path of the annotation file to read

  annotation_type : str
      The type of the annotation file that should be read. The following
      annotation_types are supported:

        * ``eyecenter``: The file contains a single row with four entries:
          ``re_x re_y le_x le_y``
        * ``named``: The file contains named annotations, one per line, e.g.:
          ``reye re_x re_y`` or ``pose 25.7``
        * ``idiap``: The file contains enumerated annotations, one per line,
          e.g.: ``1 key1_x key1_y``, and maybe some additional annotations like
          gender, age, ...


  Returns
  -------
  dict
      A python dictionary with the keypoint name as key and the
      position ``(y,x)`` as value, and maybe some additional annotations.

  Raises
  ------
  IOError
      If the annotation file is not found.
  ValueError
      If the annotation type is not known.

  """

  if not file_name:
    return None

  if not os.path.exists(file_name):
    raise IOError("The annotation file '%s' was not found" % file_name)

  annotations = {}

  with open(file_name, 'r') as f:

    if str(annotation_type) == 'eyecenter':
      # only the eye positions are written, all are in the first row
      line = f.readline()
      positions = line.split()
      assert len(positions) == 4
      annotations['reye'] = (float(positions[1]), float(positions[0]))
      annotations['leye'] = (float(positions[3]), float(positions[2]))

    elif str(annotation_type) == 'named':
      # multiple lines, no header line, each line contains annotation and
      # position or single value annotation
      for line in f:
        positions = line.split()
        if len(positions) == 3:
          annotations[positions[0]] = (
              float(positions[2]), float(positions[1]))
        elif len(positions) == 2:
          annotations[positions[0]] = float(positions[1])
        else:
          logger.error(
              "Could not interpret line '%s' in annotation file '%s'",
              line, file_name)

    elif str(annotation_type) == 'idiap':
      # Idiap format: multiple lines, no header, each line contains an integral
      # keypoint identifier, or other identifier like 'gender', 'age',...
      for line in f:
        positions = line.rstrip().split()
        if positions:
          if positions[0].isdigit():
            # position field
            assert len(positions) == 3
            id = int(positions[0])
            annotations[_idiap_annotations[id]] = (
                float(positions[2]), float(positions[1]))
          else:
            # another field, we take the first entry as key and the rest as
            # values
            annotations[positions[0]] = positions[1:]
      # finally, we add the eye center coordinates as the center between the
      # eye corners; the annotations 3 and 8 are the pupils...
      if 'reyeo' in annotations and 'reyei' in annotations:
        annotations['reye'] = ((annotations['reyeo'][0] + annotations['reyei'][0]) /
                               2., (annotations['reyeo'][1] + annotations['reyei'][1]) / 2.)
      if 'leyeo' in annotations and 'leyei' in annotations:
        annotations['leye'] = ((annotations['leyeo'][0] + annotations['leyei'][0]) /
                               2., (annotations['leyeo'][1] + annotations['leyei'][1]) / 2.)

    else:
      raise ValueError(
          "The given annotation type '%s' is not known, choose one of ('eyecenter', 'named', 'idiap')" % annotation_type)

  if 'leye' in annotations and 'reye' in annotations and annotations['leye'][1] < annotations['reye'][1]:
    logger.warn(
        "The eye annotations in file '%s' might be exchanged!" % file_name)

  return annotations
