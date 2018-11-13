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
import collections

import bob.ip.base
import bob.ip.facedetect

def scale_image(image, width, height):
  """scale_image(image, width, height) -> scaled_image
  
  This function scales an image.

  **Parameters**
  
    ``image`` : (3d numpy array)
      The image to be scaled.
  
    ``width`` : (int)
      The new image width.
  
    ``height``: (int)
      The new image height

  **Returns**
  
    ``result`` : (3d numpy array)
      The scaled image
  """
  assert len(image.shape) == 3, "This is meant to work with color images (3 channels)"
  result = numpy.zeros((3, width, height))
  bob.ip.base.scale(image, result)
  return result


def crop_face(image, bbx, facewidth):
  """crop_face(image, bbx, facewidth) -> face
  
  This function crops a face from an image.
  
  **Parameters**
  
    ``image`` : (3d numpy array )
      The image containing the face.

    ``bbx`` : (bob.ip.facedetect.BoundingBox)
      The bounding box of the face.

    ``facewidth``: (int)
      The width of the face after cropping.

  **Returns**
    
    ``face`` : (numpy array)
      The face image.
  """
  face = image[:, bbx.topleft[0]:(bbx.topleft[0] + bbx.size[0]), bbx.topleft[1]:(bbx.topleft[1] + bbx.size[1])]
  aspect_ratio = bbx.size_f[0] / bbx.size_f[1] # height/width
  faceheight = int(facewidth * aspect_ratio)
  face = scale_image(face, faceheight, facewidth)
  face = face.astype('uint8')
  return face


def build_bandpass_filter(fs, order, plot=False):
  """build_bandpass_filter(fs, order[, plot]) -> b
  
  Builds a butterworth bandpass filter.
  
  **Parameters**

    ``fs`` : (float)
      sampling frequency of the signal (i.e. framerate).
    
    ``order`` : (int)
      The order of the filter (the higher, the sharper).
  
    ``plot`` : ([Optional] boolean)
      Plots the frequency response of the filter.
      Defaults to False.
  
  **Returns**
    
    ``b`` : (numpy array)
      The coefficients of the FIR filter.
  """
  # frequency range in Hertz, corresponds to plausible h
  #heart-rate values, i.e. [42-240] beats per minute
  min_freq = 0.7 
  max_freq = 4.0 

  from scipy.signal import firwin 
  nyq = fs / 2.0
  numtaps = order + 1
  b = firwin(numtaps, [min_freq/nyq, max_freq/nyq], pass_zero=False)

  # show the frequency response of the filter
  if plot:
    from matplotlib import pyplot
    from scipy.signal import freqz
    w, h = freqz(b)
    fig = pyplot.figure()
    pyplot.title('Bandpass filter frequency response')
    ax1 = fig.add_subplot(111)
    pyplot.plot(w * fs / (2 * numpy.pi), 20 * numpy.log10(abs(h)), 'b')
    [ymin, ymax] = ax1.get_ylim()
    pyplot.vlines(min_freq, ymin, ymax, color='red', linewidths='2')
    pyplot.vlines(max_freq, ymin, ymax, color='red', linewidths='2')
    pyplot.ylabel('Amplitude [dB]', color='b')
    pyplot.xlabel('Frequency [Hz]')
    pyplot.show()

  return b
