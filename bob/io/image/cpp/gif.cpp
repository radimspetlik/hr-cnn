/**
 * @file io/cxx/ImageGifFile.cc
 * @date Fri Nov 23 16:53:00 2012 +0200
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 * @author Manuel Gunther <siebenkopf@googlemail.com>
 *
 * @brief Implements an image format reader/writer using giflib.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 * Copyright (c) 2016, Regents of the University of Colorado on behalf of the University of Colorado Colorado Springs.
 */

#ifdef HAVE_GIFLIB

#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <string>

#include <bob.io.image/gif.h>

extern "C" {
#include <gif_lib.h>
}

// QuantizeBuffer function definition that was inlined (only) in giflib 4.2
#if defined(GIFLIB_MAJOR) && GIFLIB_MAJOR < 5

#define ABS(x)    ((x) > 0 ? (x) : (-(x)))
#define COLOR_ARRAY_SIZE 32768
#define BITS_PER_PRIM_COLOR 5
#define MAX_PRIM_COLOR 0x1f

static int SortRGBAxis;

typedef struct QuantizedColorType {
  GifByteType RGB[3];
  GifByteType NewColorIndex;
  long Count;
  struct QuantizedColorType *Pnext;
} QuantizedColorType;

typedef struct NewColorMapType {
  GifByteType RGBMin[3], RGBWidth[3];
  unsigned int NumEntries; /* # of QuantizedColorType in linked list below */
  unsigned long Count; /* Total number of pixels in all the entries */
  QuantizedColorType *QuantizedColors;
} NewColorMapType;

// Routine called by qsort to compare two entries.
static int
SortCmpRtn(const void *Entry1, const void *Entry2)
{
  return (*((QuantizedColorType **) Entry1))->RGB[SortRGBAxis] -
    (*((QuantizedColorType **) Entry2))->RGB[SortRGBAxis];
}

// Routine to subdivide the RGB space recursively using median cut in each
// axes alternatingly until ColorMapSize different cubes exists.
// The biggest cube in one dimension is subdivide unless it has only one entry.
// Returns GIF_ERROR if failed, otherwise GIF_OK.
static int
SubdivColorMap(NewColorMapType * NewColorSubdiv,
    unsigned int ColorMapSize,
    unsigned int *NewColorMapSize) {
  int MaxSize;
  unsigned int i, j, Index = 0, NumEntries, MinColor, MaxColor;
  long Sum, Count;
  QuantizedColorType *QuantizedColor, **SortArray;
  while (ColorMapSize > *NewColorMapSize) {
    // Find candidate for subdivision:
    MaxSize = -1;
    for (i = 0; i < *NewColorMapSize; i++) {
      for (j = 0; j < 3; j++) {
        if ((((int)NewColorSubdiv[i].RGBWidth[j]) > MaxSize) &&
            (NewColorSubdiv[i].NumEntries > 1)) {
          MaxSize = NewColorSubdiv[i].RGBWidth[j];
          Index = i;
          SortRGBAxis = j;
        }
      }
    }
    if (MaxSize == -1)
      return GIF_OK;
    // Split the entry Index into two along the axis SortRGBAxis:
    // Sort all elements in that entry along the given axis and split at
    // the median.
    SortArray = (QuantizedColorType **)malloc(
        sizeof(QuantizedColorType *) *
        NewColorSubdiv[Index].NumEntries);
    if (SortArray == NULL)
      return GIF_ERROR;
    for (j = 0, QuantizedColor = NewColorSubdiv[Index].QuantizedColors;
        j < NewColorSubdiv[Index].NumEntries && QuantizedColor != NULL;
        j++, QuantizedColor = QuantizedColor->Pnext)
      SortArray[j] = QuantizedColor;
    qsort(SortArray, NewColorSubdiv[Index].NumEntries,
        sizeof(QuantizedColorType *), SortCmpRtn);
    // Relink the sorted list into one:
    for (j = 0; j < NewColorSubdiv[Index].NumEntries - 1; j++)
      SortArray[j]->Pnext = SortArray[j + 1];
    SortArray[NewColorSubdiv[Index].NumEntries - 1]->Pnext = NULL;
    NewColorSubdiv[Index].QuantizedColors = QuantizedColor = SortArray[0];
    free((char *)SortArray);
    // Now simply add the Counts until we have half of the Count:
    Sum = NewColorSubdiv[Index].Count / 2 - QuantizedColor->Count;
    NumEntries = 1;
    Count = QuantizedColor->Count;
    while (QuantizedColor->Pnext != NULL &&
        (Sum -= QuantizedColor->Pnext->Count) >= 0 &&
        QuantizedColor->Pnext->Pnext != NULL) {
      QuantizedColor = QuantizedColor->Pnext;
      NumEntries++;
      Count += QuantizedColor->Count;
    }
    // Save the values of the last color of the first half, and first
    // of the second half so we can update the Bounding Boxes later.
    // Also as the colors are quantized and the BBoxes are full 0..255,
    // they need to be rescaled.
    MaxColor = QuantizedColor->RGB[SortRGBAxis]; // Max. of first half
    // coverity[var_deref_op]
    MinColor = QuantizedColor->Pnext->RGB[SortRGBAxis]; // of second
    MaxColor <<= (8 - BITS_PER_PRIM_COLOR);
    MinColor <<= (8 - BITS_PER_PRIM_COLOR);
    // Partition right here:
    NewColorSubdiv[*NewColorMapSize].QuantizedColors =
      QuantizedColor->Pnext;
    QuantizedColor->Pnext = NULL;
    NewColorSubdiv[*NewColorMapSize].Count = Count;
    NewColorSubdiv[Index].Count -= Count;
    NewColorSubdiv[*NewColorMapSize].NumEntries =
      NewColorSubdiv[Index].NumEntries - NumEntries;
    NewColorSubdiv[Index].NumEntries = NumEntries;
    for (j = 0; j < 3; j++) {
      NewColorSubdiv[*NewColorMapSize].RGBMin[j] =
        NewColorSubdiv[Index].RGBMin[j];
      NewColorSubdiv[*NewColorMapSize].RGBWidth[j] =
        NewColorSubdiv[Index].RGBWidth[j];
    }
    NewColorSubdiv[*NewColorMapSize].RGBWidth[SortRGBAxis] =
      NewColorSubdiv[*NewColorMapSize].RGBMin[SortRGBAxis] +
      NewColorSubdiv[*NewColorMapSize].RGBWidth[SortRGBAxis] - MinColor;
    NewColorSubdiv[*NewColorMapSize].RGBMin[SortRGBAxis] = MinColor;
    NewColorSubdiv[Index].RGBWidth[SortRGBAxis] =
      MaxColor - NewColorSubdiv[Index].RGBMin[SortRGBAxis];
    (*NewColorMapSize)++;
  }
  return GIF_OK;
}

// Quantize high resolution image into lower one. Input image consists of a
// 2D array for each of the RGB colors with size Width by Height. There is no
// Color map for the input. Output is a quantized image with 2D array of
// indexes into the output color map.
// Note input image can be 24 bits at the most (8 for red/green/blue) and
// the output has 256 colors at the most (256 entries in the color map.).
// ColorMapSize specifies size of color map up to 256 and will be updated to
// real size before returning.
// Also non of the parameter are allocated by this routine.
// This function returns GIF_OK if succesfull, GIF_ERROR otherwise.
static int
QuantizeBuffer(unsigned int Width, unsigned int Height, int *ColorMapSize,
  GifByteType * RedInput, GifByteType * GreenInput, GifByteType * BlueInput,
  GifByteType * OutputBuffer, GifColorType * OutputColorMap)
{
  unsigned int Index, NumOfEntries;
  int i, j, MaxRGBError[3];
  unsigned int NewColorMapSize;
  long Red, Green, Blue;
  NewColorMapType NewColorSubdiv[256];
  QuantizedColorType *ColorArrayEntries, *QuantizedColor;
  ColorArrayEntries = (QuantizedColorType *)malloc(
      sizeof(QuantizedColorType) * COLOR_ARRAY_SIZE);
  if (ColorArrayEntries == NULL) {
    return GIF_ERROR;
  }
  for (i = 0; i < COLOR_ARRAY_SIZE; i++) {
    ColorArrayEntries[i].RGB[0] = i >> (2 * BITS_PER_PRIM_COLOR);
    ColorArrayEntries[i].RGB[1] = (i >> BITS_PER_PRIM_COLOR) &
      MAX_PRIM_COLOR;
    ColorArrayEntries[i].RGB[2] = i & MAX_PRIM_COLOR;
    ColorArrayEntries[i].Count = 0;
  }
  // Sample the colors and their distribution:
  for (i = 0; i < (int)(Width * Height); i++) {
    Index = ((RedInput[i] >> (8 - BITS_PER_PRIM_COLOR)) <<
        (2 * BITS_PER_PRIM_COLOR)) +
      ((GreenInput[i] >> (8 - BITS_PER_PRIM_COLOR)) <<
       BITS_PER_PRIM_COLOR) +
      (BlueInput[i] >> (8 - BITS_PER_PRIM_COLOR));
    ColorArrayEntries[Index].Count++;
  }
  // Put all the colors in the first entry of the color map, and call the
  // recursive subdivision process.
  for (i = 0; i < 256; i++) {
    NewColorSubdiv[i].QuantizedColors = NULL;
    NewColorSubdiv[i].Count = NewColorSubdiv[i].NumEntries = 0;
    for (j = 0; j < 3; j++) {
      NewColorSubdiv[i].RGBMin[j] = 0;
      NewColorSubdiv[i].RGBWidth[j] = 255;
    }
  }
  // Find the non empty entries in the color table and chain them:
  for (i = 0; i < COLOR_ARRAY_SIZE; i++)
    if (ColorArrayEntries[i].Count > 0)
      break;
  QuantizedColor = NewColorSubdiv[0].QuantizedColors = &ColorArrayEntries[i];
  NumOfEntries = 1;
  while (++i < COLOR_ARRAY_SIZE)
    if (ColorArrayEntries[i].Count > 0) {
      QuantizedColor->Pnext = &ColorArrayEntries[i];
      QuantizedColor = &ColorArrayEntries[i];
      NumOfEntries++;
    }
  QuantizedColor->Pnext = NULL;
  NewColorSubdiv[0].NumEntries = NumOfEntries; // Different sampled colors
  NewColorSubdiv[0].Count = ((long)Width) * Height; // Pixels
  NewColorMapSize = 1;
  if (SubdivColorMap(NewColorSubdiv, *ColorMapSize, &NewColorMapSize) !=
      GIF_OK) {
    free((char *)ColorArrayEntries);
    return GIF_ERROR;
  }
  if (NewColorMapSize < *ColorMapSize) {
    // And clear rest of color map:
    for (i = NewColorMapSize; i < *ColorMapSize; i++)
      OutputColorMap[i].Red = OutputColorMap[i].Green =
        OutputColorMap[i].Blue = 0;
  }
  // Average the colors in each entry to be the color to be used in the
  // output color map, and plug it into the output color map itself.
  for (i = 0; i < NewColorMapSize; i++) {
    if ((j = NewColorSubdiv[i].NumEntries) > 0) {
      QuantizedColor = NewColorSubdiv[i].QuantizedColors;
      Red = Green = Blue = 0;
      while (QuantizedColor) {
        QuantizedColor->NewColorIndex = i;
        Red += QuantizedColor->RGB[0];
        Green += QuantizedColor->RGB[1];
        Blue += QuantizedColor->RGB[2];
        QuantizedColor = QuantizedColor->Pnext;
      }
      OutputColorMap[i].Red = (Red << (8 - BITS_PER_PRIM_COLOR)) / j;
      OutputColorMap[i].Green = (Green << (8 - BITS_PER_PRIM_COLOR)) / j;
      OutputColorMap[i].Blue = (Blue << (8 - BITS_PER_PRIM_COLOR)) / j;
    } else
      fprintf(stderr,
          "\n: Null entry in quantized color map - that's weird.\n");
  }
  // Finally scan the input buffer again and put the mapped index in the
  // output buffer.
  MaxRGBError[0] = MaxRGBError[1] = MaxRGBError[2] = 0;
  for (i = 0; i < (int)(Width * Height); i++) {
    Index = ((RedInput[i] >> (8 - BITS_PER_PRIM_COLOR)) <<
        (2 * BITS_PER_PRIM_COLOR)) +
      ((GreenInput[i] >> (8 - BITS_PER_PRIM_COLOR)) <<
       BITS_PER_PRIM_COLOR) +
      (BlueInput[i] >> (8 - BITS_PER_PRIM_COLOR));
    Index = ColorArrayEntries[Index].NewColorIndex;
    OutputBuffer[i] = Index;
    if (MaxRGBError[0] < ABS(OutputColorMap[Index].Red - RedInput[i]))
      MaxRGBError[0] = ABS(OutputColorMap[Index].Red - RedInput[i]);
    if (MaxRGBError[1] < ABS(OutputColorMap[Index].Green - GreenInput[i]))
      MaxRGBError[1] = ABS(OutputColorMap[Index].Green - GreenInput[i]);
    if (MaxRGBError[2] < ABS(OutputColorMap[Index].Blue - BlueInput[i]))
      MaxRGBError[2] = ABS(OutputColorMap[Index].Blue - BlueInput[i]);
  }
  free((char *)ColorArrayEntries);
  *ColorMapSize = NewColorMapSize;
  return GIF_OK;
}

#undef ABS
#undef COLOR_ARRAY_SIZE
#undef BITS_PER_PRIM_COLOR
#undef MAX_PRIM_COLOR
#endif // End of ugly QuantizeBuffer definition for giflib 4.2

static void GifErrorHandler(const char* fname, int error) {
#if defined(GIF_LIB_VERSION) || (GIFLIB_MAJOR < 5)
  const char* error_string = "unknown error (giflib < 5)";
#else
  const char* error_string = GifErrorString(error);
#endif
  boost::format m("GIF: error in %s(): (%d) %s");
  m % fname % error;
  if (error_string) m % error_string;
  else m % "unknown error";
  throw std::runtime_error(m.str());
}

static int DGifDeleter (GifFileType* ptr) {
#if defined(GIF_LIB_VERSION) || (GIFLIB_MAJOR < 5) || (GIFLIB_MAJOR == 5) && (GIFLIB_MINOR < 1)
  return DGifCloseFile(ptr);
#else
  int error = GIF_OK;
  int retval = DGifCloseFile(ptr, &error);
  if (retval == GIF_ERROR) {
    //do not call GifErrorHandler here, or the interpreter will crash
    const char* error_string = GifErrorString(retval);
    boost::format m("In DGifCloseFile(): (%d) %s");
    m % error;
    if (error_string) m % error_string;
    else m % "unknown error";
    std::cerr << "ERROR: " << m.str() << std::endl;
  }
  return retval;
#endif
}

static boost::shared_ptr<GifFileType> make_dfile(const char *filename)
{
#if defined(GIF_LIB_VERSION) || (GIFLIB_MAJOR < 5)
  GifFileType* fp = DGifOpenFileName(filename);
  if (!fp) {
    boost::format m("cannot open file `%s'");
    m % filename;
    throw std::runtime_error(m.str());
  }
#else
  int error = GIF_OK;
  GifFileType* fp = DGifOpenFileName(filename, &error);
  if (!fp) GifErrorHandler("DGifOpenFileName", error);
#endif
  return boost::shared_ptr<GifFileType>(fp, DGifDeleter);
}


static int EGifDeleter (GifFileType* ptr) {
#if defined(GIF_LIB_VERSION) || (GIFLIB_MAJOR < 5) || (GIFLIB_MAJOR == 5) && (GIFLIB_MINOR < 1)
  return EGifCloseFile(ptr);
#else
  int error = GIF_OK;
  int retval = EGifCloseFile(ptr, &error);
  if (retval == GIF_ERROR) {
    //do not call GifErrorHandler here, or the interpreter will crash
    const char* error_string = GifErrorString(error);
    boost::format m("In EGifCloseFile(): (%d) %s");
    m % error;
    if (error_string) m % error_string;
    else m % "unknown error";
    std::cerr << "ERROR: " << m.str() << std::endl;
  }
  return retval;
#endif
}

static boost::shared_ptr<GifFileType> make_efile(const char *filename)
{
#if defined(GIF_LIB_VERSION) || (GIFLIB_MAJOR < 5)
  GifFileType* fp = EGifOpenFileName(filename, false);
  if (!fp) {
    boost::format m("cannot open file `%s'");
    m % filename;
    throw std::runtime_error(m.str());
  }
#else
  int error = GIF_OK;
  GifFileType* fp = EGifOpenFileName(filename, false, &error);
  if (!fp) GifErrorHandler("EGifOpenFileName", error);
#endif
  return boost::shared_ptr<GifFileType>(fp, EGifDeleter);
}

/**
 * LOADING
 */
static void im_peek(const std::string& path, bob::io::base::array::typeinfo& info)
{
  // 1. GIF file opening
  boost::shared_ptr<GifFileType> in_file = make_dfile(path.c_str());

  // 2. Set typeinfo variables
  info.dtype = bob::io::base::array::t_uint8;
  info.nd = 3;
  info.shape[0] = 3;
  info.shape[1] = in_file->SHeight;
  info.shape[2] = in_file->SWidth;
  info.update_strides();
}

static void im_load_color(boost::shared_ptr<GifFileType> in_file, bob::io::base::array::interface& b)
{
  const bob::io::base::array::typeinfo& info = b.type();
  const size_t height0 = info.shape[1];
  const size_t width0 = info.shape[2];
  const size_t frame_size = height0*width0;

  // The following piece of code is based on the giflib utility called gif2rgb
  // Allocate the screen as vector of column of rows. Note this
  // screen is device independent - it's the screen defined by the
  // GIF file parameters.
  std::vector<boost::shared_array<GifPixelType> > screen_buffer;

  // Size in bytes one row.
  int size = in_file->SWidth*sizeof(GifPixelType);
  // First row
  screen_buffer.push_back(boost::shared_array<GifPixelType>(new GifPixelType[in_file->SWidth]));

  // Set its color to BackGround
  for(int i=0; i<in_file->SWidth; ++i)
    screen_buffer[0][i] = in_file->SBackGroundColor;
  for(int i=1; i<in_file->SHeight; ++i) {
    // Allocate the other rows, and set their color to background too:
    screen_buffer.push_back(boost::shared_array<GifPixelType>(new GifPixelType[in_file->SWidth]));
    memcpy(screen_buffer[i].get(), screen_buffer[0].get(), size);
  }

  // Scan the content of the GIF file and load the image(s) in:
  GifRecordType record_type;
  GifByteType *extension;
  int InterlacedOffset[] = { 0, 4, 2, 1 }; // The way Interlaced image should.
  int InterlacedJumps[] = { 8, 8, 4, 2 }; // be read - offsets and jumps...
  int row, col, width, height, count, ext_code;
  int error = DGifGetRecordType(in_file.get(), &record_type);
  if(error == GIF_ERROR)
    GifErrorHandler("DGifGetRecordType", error);
  switch(record_type) {
    case IMAGE_DESC_RECORD_TYPE:
      error = DGifGetImageDesc(in_file.get());
      if (error == GIF_ERROR) GifErrorHandler("DGifGetImageDesc", error);
      row = in_file->Image.Top; // Image Position relative to Screen.
      col = in_file->Image.Left;
      width = in_file->Image.Width;
      height = in_file->Image.Height;
      if(in_file->Image.Left + in_file->Image.Width > in_file->SWidth ||
        in_file->Image.Top + in_file->Image.Height > in_file->SHeight)
      {
        throw std::runtime_error("GIF: the dimensions of image larger than the dimensions of the canvas.");
      }
      if(in_file->Image.Interlace) {
        // Need to perform 4 passes on the images:
        for(int i=count=0; i<4; ++i)
          for(int j=row+InterlacedOffset[i]; j<row+height; j+=InterlacedJumps[i]) {
            ++count;
            error = DGifGetLine(in_file.get(), &screen_buffer[j][col], width);
            if(error == GIF_ERROR) GifErrorHandler("DGifGetLine", error);
          }
      }
      else {
        for(int i=0; i<height; ++i) {
          error = DGifGetLine(in_file.get(), &screen_buffer[row++][col], width);
          if(error == GIF_ERROR) GifErrorHandler("DGifGetLine", error);
        }
      }
      break;
    case EXTENSION_RECORD_TYPE:
      // Skip any extension blocks in file:
      error = DGifGetExtension(in_file.get(), &ext_code, &extension);
      if (error == GIF_ERROR) GifErrorHandler("DGifGetExtension", error);
      while(extension != NULL) {
        error = DGifGetExtensionNext(in_file.get(), &extension);
        if(error == GIF_ERROR) GifErrorHandler("DGifGetExtensionNext", error);
      }
      break;
    case TERMINATE_RECORD_TYPE:
      break;
    default: // Should be trapped by DGifGetRecordType.
      break;
  }

  // Lets dump it - set the global variables required and do it:
  ColorMapObject *ColorMap = (in_file->Image.ColorMap ? in_file->Image.ColorMap : in_file->SColorMap);
  if(ColorMap == 0)
    throw std::runtime_error("GIF: image does not have a colormap");

  // Put data into C-style buffer
  uint8_t *element_r = reinterpret_cast<uint8_t*>(b.ptr());
  uint8_t *element_g = element_r + frame_size;
  uint8_t *element_b = element_g + frame_size;
  GifRowType gif_row;
  GifColorType *ColorMapEntry;
  for(int i=0; i<in_file->SHeight; ++i) {
    gif_row = screen_buffer[i].get();
    for(int j=0; j<in_file->SWidth; ++j) {
      ColorMapEntry = &ColorMap->Colors[gif_row[j]];
      *element_r++ = ColorMapEntry->Red;
      *element_g++ = ColorMapEntry->Green;
      *element_b++ = ColorMapEntry->Blue;
    }
  }
}

static void im_load(const std::string& filename, bob::io::base::array::interface& b)
{
  // 1. GIF file opening
  boost::shared_ptr<GifFileType> in_file = make_dfile(filename.c_str());

  // 2. Read content
  const bob::io::base::array::typeinfo& info = b.type();
  if (info.dtype == bob::io::base::array::t_uint8) {
    if (info.nd == 3) im_load_color(in_file, b);
    else {
      boost::format m("GIF: cannot read object of type `%s' from file `%s'");
      m % info.str() % filename;
      throw std::runtime_error(m.str());
    }
  }
  else {
    boost::format m("GIF: cannot read object of type `%s' from file `%s'");
    m % info.str() % filename;
    throw std::runtime_error(m.str());
  }
}

/**
 * SAVING
 */
static void im_save_color(const bob::io::base::array::interface& b, boost::shared_ptr<GifFileType> out_file)
{
  const bob::io::base::array::typeinfo& info = b.type();
  const int height = info.shape[1];
  const int width = info.shape[2];
  const size_t frame_size = height * width;

  // pointer to a single row (tiff_bytep is a typedef to unsigned char or char)
  const uint8_t *element_r = static_cast<const uint8_t*>(b.ptr());
  const uint8_t *element_g = element_r + frame_size;
  const uint8_t *element_b = element_g + frame_size;

  GifByteType *red_buffer = const_cast<GifByteType*>(reinterpret_cast<const GifByteType*>(element_r));
  GifByteType *green_buffer = const_cast<GifByteType*>(reinterpret_cast<const GifByteType*>(element_g));
  GifByteType *blue_buffer = const_cast<GifByteType*>(reinterpret_cast<const GifByteType*>(element_b));
  boost::shared_array<GifByteType> output_buffer(new GifByteType[width*height]);

  // The following piece of code is based on the giflib utility called gif2rgb
  const int ExpNumOfColors = 8;
  int ColorMapSize = 1 << ExpNumOfColors;
  ColorMapObject *OutputColorMap = 0;

#if defined(GIF_LIB_VERSION) || (GIFLIB_MAJOR < 5)
  if((OutputColorMap = MakeMapObject(ColorMapSize, NULL)) == 0)
#else
  if((OutputColorMap = GifMakeMapObject(ColorMapSize, NULL)) == 0)
#endif
    throw std::runtime_error("GIF: error in GifMakeMapObject().");

  int error;
#if defined(GIF_LIB_VERSION) || (GIFLIB_MAJOR < 5)
  error = QuantizeBuffer(width, height, &ColorMapSize,
      red_buffer, green_buffer, blue_buffer, output_buffer.get(),
      OutputColorMap->Colors);
#else
  error = GifQuantizeBuffer(width, height, &ColorMapSize,
      red_buffer, green_buffer, blue_buffer, output_buffer.get(),
      OutputColorMap->Colors);
#endif
  if (error == GIF_ERROR) GifErrorHandler("GifQuantizeBuffer", error);

  error = EGifPutScreenDesc(out_file.get(), width, height, ExpNumOfColors, 0,
      OutputColorMap);
  if (error == GIF_ERROR) GifErrorHandler("EGifPutScreenDesc", error);

  error = EGifPutImageDesc(out_file.get(), 0, 0, width, height, false, NULL);
  if (error == GIF_ERROR) GifErrorHandler("EGifPutImageDesc", error);

  GifByteType *ptr = output_buffer.get();
  for(int i=0; i<height; ++i) {
    error = EGifPutLine(out_file.get(), ptr, width);
    if (error == GIF_ERROR) GifErrorHandler("EGifPutImageDesc", error);
    ptr += width;
  }

  // Free map object
#if defined(GIF_LIB_VERSION) || (GIFLIB_MAJOR < 5)
  FreeMapObject(OutputColorMap);
#else
  GifFreeMapObject(OutputColorMap);
#endif
}

static void im_save(const std::string& filename, const bob::io::base::array::interface& array)
{
  // 1. GIF file opening
  boost::shared_ptr<GifFileType> out_file = make_efile(filename.c_str());

  // 2. Set the image information here:
  const bob::io::base::array::typeinfo& info = array.type();

  // 3. Writes content
  if(info.dtype == bob::io::base::array::t_uint8) {
    if(info.nd == 3) {
      if(info.shape[0] != 3)
        throw std::runtime_error("color image does not have 3 planes on 1st. dimension");
      im_save_color(array, out_file);
    }
    else {
      boost::format m("GIF: cannot save object of type `%s' to file `%s'");
      m % info.str() % filename;
      throw std::runtime_error(m.str());
    }
  }
  else {
    boost::format m("GIF: cannot save object of type `%s' to file `%s'");
    m % info.str() % filename;
    throw std::runtime_error(m.str());
  }
}


/**
 * GIF class
*/
bob::io::image::GIFFile::GIFFile(const char* path, char mode)
: m_filename(path),
  m_newfile(true) {

  //checks if file exists
  if (mode == 'r' && !boost::filesystem::exists(path)) {
    boost::format m("file '%s' is not readable");
    m % path;
    throw std::runtime_error(m.str());
  }

  if (mode == 'r' || (mode == 'a' && boost::filesystem::exists(path))) {
    im_peek(path, m_type);
    m_length = 1;
    m_newfile = false;
  }
  else {
    m_length = 0;
    m_newfile = true;
  }
}

void bob::io::image::GIFFile::read(bob::io::base::array::interface& buffer, size_t index) {
  if (m_newfile)
    throw std::runtime_error("uninitialized image file cannot be read");

  if (!buffer.type().is_compatible(m_type)) buffer.set(m_type);

  if (index != 0)
    throw std::runtime_error("cannot read image with index > 0 -- there is only one image in an image file");

  if(!buffer.type().is_compatible(m_type)) buffer.set(m_type);
  im_load(m_filename, buffer);
}

size_t bob::io::image::GIFFile::append(const bob::io::base::array::interface& buffer) {
  if (m_newfile) {
    im_save(m_filename, buffer);
    m_type = buffer.type();
    m_newfile = false;
    m_length = 1;
    return 0;
  }

  throw std::runtime_error("image files only accept a single array");
}

void bob::io::image::GIFFile::write (const bob::io::base::array::interface& buffer) {
  //overwriting position 0 should always work
  if (m_newfile) {
    append(buffer);
    return;
  }

  throw std::runtime_error("image files only accept a single array");
}

std::string bob::io::image::GIFFile::s_codecname = "bob.image_gif";

boost::shared_ptr<bob::io::base::File> make_gif_file (const char* path, char mode) {
  return boost::make_shared<bob::io::image::GIFFile>(path, mode);
}

#endif // HAVE_GIFLIB
