#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Guillaume Heusch <guillaume.heusch@idiap.ch>,
# 
# This file is part of bob.rpgg.base.
# 
# bob.rppg.base is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# bob.rppg.base is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with bob.rppg.base. If not, see <http://www.gnu.org/licenses/>.

import os, sys
import numpy

def test_scale_image():
  """
  Tests the rescaling of an image
  """

  from bob.rppg.base.utils import scale_image
  image = numpy.zeros((3, 100, 100), dtype='uint8')
  scaled = scale_image(image, 50, 75)
  assert scaled.shape == (3, 50, 75)

def test_crop_face():
  """
  Test the cropping of a face
  """
  image = numpy.zeros((3, 100, 100), dtype='uint8')
  
  from bob.ip.facedetect import BoundingBox
  bbox = BoundingBox((20, 20), (50, 50))

  from bob.rppg.base.utils import crop_face
  cropped = crop_face(image, bbox, 48)
  assert cropped.shape == (3, 48, 48)

