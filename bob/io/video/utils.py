#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Thu 14 Mar 17:00:58 2013
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Some utilities to generate fake patterns
"""

import numpy

DEFAULT_FONT = __import__('pkg_resources').resource_filename(__name__,
    __import__('os').path.join("fonts", "regular.ttf"))

def estimate_fontsize(height, width, format):
  """Estimates the best fontsize to fit into a image that is (height, width)"""

  try:
    # if PIL is installed this works:
    import Image, ImageFont, ImageDraw
  except ImportError:
    # if Pillow is installed, this works better:
    from PIL import Image, ImageFont, ImageDraw

  best_size = min(height, width)
  fit = False
  while best_size > 0:
    font = ImageFont.truetype(DEFAULT_FONT, best_size)
    (text_width, text_height) = font.getsize(format % 0)
    if text_width < width and text_height < height: break
    best_size -= 1

  if best_size <= 0:
    raise RuntimeError("Cannot find best size for font")

  return best_size

def print_numbers(frame, counter, format, fontsize):
  """Generates an image that serves as a test pattern for encoding/decoding and
  accuracy tests."""

  try:
    # if PIL is installed this works:
    import Image, ImageFont, ImageDraw
  except ImportError:
    # if Pillow is installed, this works better:
    from PIL import Image, ImageFont, ImageDraw

  _, height, width = frame.shape

  # text at the center, indicating the frame number
  text = format % counter
  dim = min(width, height)
  font = ImageFont.truetype(DEFAULT_FONT, fontsize)
  (text_width, text_height) = font.getsize(text)
  x_pos = int((width - text_width) / 2)
  y_pos = int((height - text_height) / 2)
  # this is buggy in Pillow-2.0.0, so we do it manually
  #img = Image.fromarray(frame.transpose(1,2,0))
  # replace fromstring() to frombytes() as fromstring is depricated in Pillow since version 2.0
  # img = Image.fromstring('RGB', (frame.shape[1], frame.shape[2]), frame.transpose(1,2,0).tostring())
  # img = Image.frombytes('RGB', (frame.shape[1], frame.shape[2]), frame.transpose(1,2,0).tostring()) #For some reason there is no frombytes in UBUNTU 12 04
  img = Image.frombuffer('RGB', (frame.shape[1], frame.shape[2]), frame.transpose(1,2,0).tostring(), 'raw', "RGB", 0,1 ) #This call seems weird, but I follow the instructions from here (http://pillow.readthedocs.org/en/3.0.x/reference/Image.html#PIL.Image.frombuffer). Following these instructions I don't get a warning

  draw = ImageDraw.Draw(img)
  draw.text((x_pos, y_pos), text, font=font, fill=(255,255,255))
  return numpy.asarray(img).transpose(2,0,1)

def generate_colors(height, width, shift):
  """Generates an image that serves as a test pattern for encoding/decoding and
  accuracy tests."""

  retval = numpy.ndarray((3, height, width), dtype='uint8')

  # standard color test pattern
  w = width // 7; w2 = 2*w; w3 = 3*w; w4 = 4*w; w5 = 5*w; w6 = 6*w
  retval[0,:,0:w]   = 255; retval[1,:,0:w]   = 255; retval[2,:,0:w]   = 255;
  retval[0,:,w:w2]  = 255; retval[1,:,w:w2]  = 255; retval[2,:,w:w2]  = 0;
  retval[0,:,w2:w3] = 0;   retval[1,:,w2:w3] = 255; retval[2,:,w2:w3] = 255;
  retval[0,:,w3:w4] = 0;   retval[1,:,w3:w4] = 255; retval[2,:,w3:w4] = 0;
  retval[0,:,w4:w5] = 255; retval[1,:,w4:w5] = 0;   retval[2,:,w4:w5] = 255;
  retval[0,:,w5:w6] = 255; retval[1,:,w5:w6] = 0;   retval[2,:,w5:w6] = 0;
  retval[0,:,w6:]   = 0;   retval[1,:,w6:]  = 0;   retval[2,:,w6:]   = 255;

  # rotate by 'shift'
  retval = numpy.roll(retval, shift, axis=2)
  return retval

def color_distortion(shape, framerate, format, codec, filename):
  """Returns distortion patterns for a set of frames with moving colors.

  Keyword parameters:

  shape (int, int, int)
    The length (number of frames), height and width for the generated sequence

  format
    The string that identifies the format to be used for the output file

  codec
    The codec to be used for the output file

  filename
    The name of the file to use for encoding the test
  """

  length, height, width = shape
  from . import reader, writer
  outv = writer(filename, height, width, framerate, codec=codec,
      format=format, check=True)
  orig = []
  text_format = "%%0%dd" % len(str(length-1))
  fontsize = estimate_fontsize(height, width, text_format)
  fontsize = int(fontsize/4)
  for i in range(0, length):
    newframe = generate_colors(height, width, i%width)
    newframe = print_numbers(newframe, i, text_format, fontsize)
    outv.append(newframe)
    orig.append(newframe)
  outv.close()
  orig = numpy.array(orig, dtype='uint8')
  return orig, framerate, reader(filename, check=True)

def frameskip_detection(shape, framerate, format, codec, filename):
  """Returns distortion patterns for a set of frames with big numbers.

  Keyword parameters:

  shape (int, int, int)
    The length (number of frames), height and width for the generated sequence

  format
    The string that identifies the format to be used for the output file

  codec
    The codec to be used for the output file

  filename
    The name of the file to use for encoding the test
  """

  length, height, width = shape
  from . import reader, writer
  text_format = "%%0%dd" % len(str(length-1))
  fontsize = estimate_fontsize(height, width, text_format)
  outv = writer(filename, height, width, framerate, codec=codec,
      format=format, check=True)
  orig = []
  for i in range(0, length):
    newframe = numpy.zeros((3, height, width), dtype='uint8')
    newframe = print_numbers(newframe, i, text_format, fontsize)
    outv.append(newframe)
    orig.append(newframe)
  outv.close()
  orig = numpy.array(orig, dtype='uint8')
  return orig, framerate, reader(filename, check=True)

def quality_degradation(shape, framerate, format, codec, filename):
  """Returns noise patterns for a set of frames.

  Keyword parameters:

  shape (int, int, int)
    The length (number of frames), height and width for the generated sequence

  format
    The string that identifies the format to be used for the output file

  codec
    The codec to be used for the output file

  filename
    The name of the file to use for encoding the test
  """

  length, height, width = shape
  from . import reader, writer
  text_format = "%%0%dd" % len(str(length-1))
  fontsize = estimate_fontsize(height, width, text_format)
  fontsize = int(fontsize/4)
  outv = writer(filename, height, width, framerate, codec=codec,
      format=format, check=True)
  orig = []
  for i in range(0, length):
    newframe = numpy.random.randint(0, 256, (3, height, width)).astype('uint8')
    newframe = print_numbers(newframe, i, text_format, fontsize)
    outv.append(newframe)
    orig.append(newframe)
  outv.close()
  orig = numpy.array(orig, dtype='uint8')
  return orig, framerate, reader(filename, check=True)

def is_string(s):
  """Returns ``True`` if the given object is a string

  This method can be used with Python-2.x or 3.x and returns a string
  respecting each environment's constraints.
  """

  from sys import version_info

  return (version_info[0] < 3 and isinstance(s, (str, unicode))) or \
    isinstance(s, (bytes, str))
