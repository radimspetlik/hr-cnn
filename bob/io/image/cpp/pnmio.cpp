/*
 * File       : pnmio.c
 * Description: I/O facilities for PBM, PGM, PPM (PNM) ASCII images.
 * Author     : Nikolaos Kavvadias <nikolaos.kavvadias@gmail.com>
 * Copyright  : (C) Nikolaos Kavvadias 2012, 2013, 2014, 2015, 2016
 * Website    : http://www.nkavvadias.com
 *
 * This file is part of libpnmio, and is distributed under the terms of the
 * Modified BSD License.
 *
 * A copy of the Modified BSD License is included with this distribution
 * in the file LICENSE.
 * libpnmio is free software: you can redistribute it and/or modify it under the
 * terms of the Modified BSD License.
 * libpnmio is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the Modified BSD License for more details.
 *
 * You should have received a copy of the Modified BSD License along with
 * libpnmio. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include "pnmio.h"

#define  MAXLINE         1024
// #define  LITTLE_ENDIAN     -1
// #define  BIG_ENDIAN         1
#define  GREYSCALE_TYPE     0 /* used for PFM */
#define  RGB_TYPE           1 /* used for PFM */


/* get_pnm_type:
 * Read the header contents of a PBM/PGM/PPM/PFM file up to the point of
 * extracting its type. Valid types for a PNM image are as follows:
 *   PBM_ASCII     =  1
 *   PGM_ASCII     =  2
 *   PPM_ASCII     =  3
 *   PBM_BINARY    =  4
 *   PGM_BINARY    =  5
 *   PPM_BINARY    =  6
 *   PAM           =  7
 *   PFM_RGB       = 16
 *   PFM_GREYSCALE = 17
 *
 * The result (pnm_type) is returned.
 */
int get_pnm_type(FILE *f)
{
  int flag=0;
  int pnm_type=0;
  unsigned int i;
  char magic[MAXLINE];
  char line[MAXLINE];

  /* Read the PNM/PFM file header. */
  while (fgets(line, MAXLINE, f) != NULL) {
    flag = 0;
    for (i = 0; i < strlen(line); i++) {
      if (isgraph(line[i])) {
        if ((line[i] == '#') && (flag == 0)) {
          flag = 1;
        }
      }
    }
    if (flag == 0) {
      sscanf(line, "%s", magic);
      break;
    }
  }

  /* NOTE: This part can be written more succinctly, however,
   * it is better to have the PNM types decoded explicitly.
   */
  if (strcmp(magic, "P1") == 0) {
    pnm_type = PBM_ASCII;
  } else if (strcmp(magic, "P2") == 0) {
    pnm_type = PGM_ASCII;
  } else if (strcmp(magic, "P3") == 0) {
    pnm_type = PPM_ASCII;
  } else if (strcmp(magic, "P4") == 0) {
    pnm_type = PBM_BINARY;
  } else if (strcmp(magic, "P5") == 0) {
    pnm_type = PGM_BINARY;
  } else if (strcmp(magic, "P6") == 0) {
    pnm_type = PPM_BINARY;
  } else if (strcmp(magic, "P7") == 0) {
    pnm_type = PAM;
  } else if (strcmp(magic, "PF") == 0) {
    pnm_type = PFM_RGB;
  } else if (strcmp(magic, "Pf") == 0) {
    pnm_type = PFM_GREYSCALE;
  } else {
    pnm_type = -1;
  }

  return (pnm_type);
}

/* read_pbm_header:
 * Read the header contents of a PBM (Portable Binary Map) file.
 * An ASCII PBM image file follows the format:
 * P1
 * <X> <Y>
 * <I1> <I2> ... <IMAX>
 * A binary PBM image file uses P4 instead of P1 and
 * the data values are represented in binary.
 * NOTE1: Comment lines start with '#'.
 * NOTE2: < > denote integer values (in decimal).
 */
int read_pbm_header(FILE *f, int *img_xdim, int *img_ydim, int *is_ascii)
{
  int flag=0;
  int x_val, y_val;
  unsigned int i;
  char magic[MAXLINE];
  char line[MAXLINE];
  int count=0;

  /* Read the PBM file header. */
  while (fgets(line, MAXLINE, f) != NULL) {
    flag = 0;
    for (i = 0; i < strlen(line); i++) {
      if (isgraph(line[i])) {
        if ((line[i] == '#') && (flag == 0)) {
          flag = 1;
        }
      }
    }
    if (flag == 0) {
      if (count == 0) {
        count += sscanf(line, "%s %d %d", magic, &x_val, &y_val);
      } else if (count == 1) {
        count += sscanf(line, "%d %d", &x_val, &y_val);
      } else if (count == 2) {
        count += sscanf(line, "%d", &y_val);
      }
    }
    if (count == 3) {
      break;
    }
  }

  if (strcmp(magic, "P1") == 0) {
    *is_ascii = 1;
  } else if (strcmp(magic, "P4") == 0) {
    *is_ascii = 0;
  } else {
    return -1;
    // fprintf(stderr, "Error: Input file not in PBM format!\n");
    // exit(1);
  }

  // fprintf(stderr, "Info: magic=%s, x_val=%d, y_val=%d\n",
  //   magic, x_val, y_val);
  *img_xdim   = x_val;
  *img_ydim   = y_val;
  return 0;
}

/* read_pgm_header:
 * Read the header contents of a PGM (Portable Grey[scale] Map) file.
 * An ASCII PGM image file follows the format:
 * P2
 * <X> <Y>
 * <levels>
 * <I1> <I2> ... <IMAX>
 * A binary PGM image file uses P5 instead of P2 and
 * the data values are represented in binary.
 * NOTE1: Comment lines start with '#'.
 * NOTE2: < > denote integer values (in decimal).
 */
int read_pgm_header(FILE *f, int *img_xdim, int *img_ydim, int *img_colors, int *is_ascii)
{
  int flag=0;
  int x_val, y_val, maxcolors_val;
  unsigned int i;
  char magic[MAXLINE];
  char line[MAXLINE];
  int count=0;

  /* Read the PGM file header. */
  while (fgets(line, MAXLINE, f) != NULL) {
    flag = 0;
    for (i = 0; i < strlen(line); i++) {
      if (isgraph(line[i]) && (flag == 0)) {
        if ((line[i] == '#') && (flag == 0)) {
          flag = 1;
        }
      }
    }
    if (flag == 0) {
      if (count == 0) {
        count += sscanf(line, "%s %d %d %d", magic, &x_val, &y_val, &maxcolors_val);
      } else if (count == 1) {
        count += sscanf(line, "%d %d %d", &x_val, &y_val, &maxcolors_val);
      } else if (count == 2) {
        count += sscanf(line, "%d %d", &y_val, &maxcolors_val);
      } else if (count == 3) {
        count += sscanf(line, "%d", &maxcolors_val);
      }
    }
    if (count == 4) {
      break;
    }
  }

  if (strcmp(magic, "P2") == 0) {
    *is_ascii = 1;
  } else if (strcmp(magic, "P5") == 0) {
    *is_ascii = 0;
  } else {
    return -1;
    // fprintf(stderr, "Error: Input file not in PGM format!\n");
    // exit(1);
  }

  // fprintf(stderr, "Info: magic=%s, x_val=%d, y_val=%d, maxcolors_val=%d\n",
  //   magic, x_val, y_val, maxcolors_val);
  *img_xdim   = x_val;
  *img_ydim   = y_val;
  *img_colors = maxcolors_val;
  return 0;
}

/* read_ppm_header:
 * Read the header contents of a PPM (Portable Pix[el] Map) file.
 * An ASCII PPM image file follows the format:
 * P3
 * <X> <Y>
 * <colors>
 * <R1> <G1> <B1> ... <RMAX> <GMAX> <BMAX>
 * A binary PPM image file uses P6 instead of P3 and
 * the data values are represented in binary.
 * NOTE1: Comment lines start with '#'.
 # NOTE2: < > denote integer values (in decimal).
 */
int read_ppm_header(FILE *f, int *img_xdim, int *img_ydim, int *img_colors, int *is_ascii)
{
  int flag=0;
  int x_val, y_val, maxcolors_val;
  unsigned int i;
  char magic[MAXLINE];
  char line[MAXLINE];
  int count=0;

  /* Read the PPM file header. */
  while (fgets(line, MAXLINE, f) != NULL) {
    flag = 0;
    for (i = 0; i < strlen(line); i++) {
      if (isgraph(line[i]) && (flag == 0)) {
        if ((line[i] == '#') && (flag == 0)) {
          flag = 1;
        }
      }
    }
    if (flag == 0) {
      if (count == 0) {
        count += sscanf(line, "%s %d %d %d", magic, &x_val, &y_val, &maxcolors_val);
      } else if (count == 1) {
        count += sscanf(line, "%d %d %d", &x_val, &y_val, &maxcolors_val);
      } else if (count == 2) {
        count += sscanf(line, "%d %d", &y_val, &maxcolors_val);
      } else if (count == 3) {
        count += sscanf(line, "%d", &maxcolors_val);
      }
    }
    if (count == 4) {
      break;
    }
  }

  if (strcmp(magic, "P3") == 0) {
    *is_ascii = 1;
  } else if (strcmp(magic, "P6") == 0) {
    *is_ascii = 0;
  } else {
    return -1;
    // fprintf(stderr, "Error: Input file not in PPM format!\n");
    // exit(1);
  }

  // fprintf(stderr, "Info: magic=%s, x_val=%d, y_val=%d, maxcolors_val=%d\n",
  //   magic, x_val, y_val, maxcolors_val);
  *img_xdim   = x_val;
  *img_ydim   = y_val;
  *img_colors = maxcolors_val;
  return 0;
}

/* read_pbm_data:
 * Read the data contents of a PBM (portable bit map) file.
 */
int read_pbm_data(FILE *f, int *img_in, int img_size, int is_ascii, int img_width)
{
  int i=0, c;
  int lum_val;
  int k;
  int row_position = 0;
  int read_count;

  /* Read the rest of the PPM file. */
  while ((c = fgetc(f)) != EOF) {
    ungetc(c, f);
    if (is_ascii == 1) {
      read_count = fscanf(f, "%d", &lum_val);
      if (read_count < 1) return -1;
      if (i >= img_size) break;
      img_in[i++] = lum_val;
    } else {
      lum_val = fgetc(f);
      /* Decode the image contents byte-by-byte. */
      for (k = 0; k < 8; k++) {
        if (i >= img_size) break;
        img_in[i++] = (lum_val >> (7-k)) & 0x1;
        // fprintf(stderr, "i: %d, %d\n", i, img_in[i]);
        row_position++;
        if (row_position >= img_width) {
          row_position = 0;
          break;
        }
      }
    }
  }
  // fclose(f);
  return 0;
}

/* read_pgm_data:
 * Read the data contents of a PGM (portable grey map) file.
 */
int read_pgm_data(FILE *f, int *img_in, int img_size, int is_ascii,
  unsigned int bytes_per_sample)
{
  int i=0, c;
  int lum_val;
  int read_count;

  /* Read the rest of the PPM file. */
  while ((c = fgetc(f)) != EOF) {
    ungetc(c, f);
    if (is_ascii == 1) {
	    read_count = fscanf(f, "%d", &lum_val);
      if (read_count < 1) return -1;
	  } else {
      if (bytes_per_sample == 1) {
        lum_val = fgetc(f);
      } else {
        lum_val = fgetc(f);
        lum_val = lum_val << 8;
        lum_val |= fgetc(f);
      }
    }
    if (i >= img_size) break;
    img_in[i++] = lum_val;
  }
  // fclose(f);
  return 0;
}

/* read_ppm_data:
 * Read the data contents of a PPM (portable pix map) file.
 */
int read_ppm_data(FILE *f, int *img_in, int img_size, int is_ascii,
  unsigned int bytes_per_sample)
{
  int i=0, c;
  int r_val, g_val, b_val;
  int read_count;

  /* Read the rest of the PPM file. */
  while ((c = fgetc(f)) != EOF) {
    ungetc(c, f);
    if (is_ascii == 1) {
      read_count = fscanf(f, "%d %d %d", &r_val, &g_val, &b_val);
      if (read_count < 3) return -1;
    } else {
      if (bytes_per_sample == 1) {
        r_val = fgetc(f);
        g_val = fgetc(f);
        b_val = fgetc(f);
      } else {
        r_val = fgetc(f);
        r_val = r_val << 8;
        r_val |= fgetc(f);

        g_val = fgetc(f);
        g_val = g_val << 8;
        g_val |= fgetc(f);

        b_val = fgetc(f);
        b_val = b_val << 8;
        b_val |= fgetc(f);
      }
    }
    if (i >= img_size) break;
    img_in[i++] = r_val;
    img_in[i++] = g_val;
    img_in[i++] = b_val;
  }
  // fclose(f);
  return 0;
}

/* write_pbm_file:
 * Write the contents of a PBM (portable bit map) file.
 */
int write_pbm_file(FILE *f, int *img_out,
  int x_size, int y_size, int x_scale_val, int y_scale_val, int linevals,
  int is_ascii)
{
  int i, j, x_scaled_size, y_scaled_size;
  int k, v, temp, step;
  int row_position = 0;

  x_scaled_size = x_size * x_scale_val;
  y_scaled_size = y_size * y_scale_val;
  /* Write the magic number string. */
  if (is_ascii == 1) {
    fprintf(f, "P1\n");
	step = 1;
  } else {
    fprintf(f, "P4\n");
	step = 8;
  }
  /* Write a comment containing the file name. */
  // fprintf(f, "# %s\n", img_out_fname);
  /* Write the image dimensions. */
  fprintf(f, "%d %d\n", x_scaled_size, y_scaled_size);

  /* Write the image data. */
  for (i = 0; i < y_scaled_size; i++) {
    for (j = 0; j < x_scaled_size; j+=step) {
	    if (is_ascii == 1) {
        fprintf(f, "%d ", img_out[i*x_scaled_size+j]);
	    } else {
	      temp = 0;
		    for (k = 0; k < 8; k++) {
          v = img_out[i*x_scaled_size+j+k];
          temp |= (v << (7-k));
          row_position++;
          if (row_position >= x_size) {
            row_position = 0;
            break;
          }
		    }
        fprintf(f, "%c", temp);
      }
      if (((i*x_scaled_size+j) % linevals) == (linevals-1)) {
        fprintf(f, "\n");
      }
    }
  }
  // fclose(f);
  return 0;
}

/* write_pgm_file:
 * Write the contents of a PGM (portable grey map) file.
 */
int write_pgm_file(FILE *f, int *img_out,
  int x_size, int y_size, int x_scale_val, int y_scale_val,
  int img_colors, int linevals, int is_ascii,
  unsigned int bytes_per_sample)
{
  int i, j, x_scaled_size, y_scaled_size;

  x_scaled_size = x_size * x_scale_val;
  y_scaled_size = y_size * y_scale_val;
  /* Write the magic number string. */
  if (is_ascii == 1) {
    fprintf(f, "P2\n");
  } else {
    fprintf(f, "P5\n");
  }
  /* Write a comment containing the file name. */
  // fprintf(f, "# %s\n", img_out_fname);
  /* Write the image dimensions. */
  fprintf(f, "%d %d\n", x_scaled_size, y_scaled_size);
  /* Write the maximum color/grey level allowed. */
  fprintf(f, "%d\n", img_colors);

  /* Write the image data. */
  for (i = 0; i < y_scaled_size; i++) {
    for (j = 0; j < x_scaled_size; j++) {
      if (is_ascii == 1) {
        fprintf(f, "%d ", img_out[i*x_scaled_size+j]);
        if (((i*x_scaled_size+j) % linevals) == (linevals-1)) {
          fprintf(f, "\n");
        }
      } else {
        if (bytes_per_sample == 1) {
          fprintf(f, "%c", img_out[i*x_scaled_size+j]);
        } else {
          fprintf(f, "%c", img_out[i*x_scaled_size+j]);
          fprintf(f, "%c", (img_out[i*x_scaled_size+j] >> 8));
        }
      }
    }
  }
  // fclose(f);
  return 0;
}

/* write_ppm_file:
 * Write the contents of a PPM (portable pix map) file.
 */
int write_ppm_file(FILE *f, int *img_out,
  int x_size, int y_size, int x_scale_val, int y_scale_val,
  int img_colors, int is_ascii, unsigned int bytes_per_sample)
{
  int i, j, x_scaled_size, y_scaled_size;

  x_scaled_size = x_size * x_scale_val;
  y_scaled_size = y_size * y_scale_val;
  /* Write the magic number string. */
  if (is_ascii == 1) {
    fprintf(f, "P3\n");
  } else {
    fprintf(f, "P6\n");
  }
  /* Write a comment containing the file name. */
  // fprintf(f, "# %s\n", img_out_fname);
  /* Write the image dimensions. */
  fprintf(f, "%d %d\n", x_scaled_size, y_scaled_size);
  /* Write the maximum color/grey level allowed. */
  fprintf(f, "%d\n", img_colors);

  /* Write the image data. */
  for (i = 0; i < y_scaled_size; i++) {
    for (j = 0; j < x_scaled_size; j++) {
      if (is_ascii == 1) {
        fprintf(f, "%d %d %d ",
          img_out[3*(i*x_scaled_size+j)+0],
          img_out[3*(i*x_scaled_size+j)+1],
          img_out[3*(i*x_scaled_size+j)+2]);
        if ((j % 4) == 0) {
          fprintf(f, "\n");
        }
      } else {
        if (bytes_per_sample == 1) {
          fprintf(f, "%c%c%c",
            img_out[3*(i*x_scaled_size+j)+0],
            img_out[3*(i*x_scaled_size+j)+1],
            img_out[3*(i*x_scaled_size+j)+2]);
        } else {
          fprintf(f, "%c%c%c",
            img_out[3*(i*x_scaled_size+j)+0],
            img_out[3*(i*x_scaled_size+j)+1],
            img_out[3*(i*x_scaled_size+j)+2]);
          fprintf(f, "%c%c%c",
            img_out[3*(i*x_scaled_size+j)+0] >> 8,
            img_out[3*(i*x_scaled_size+j)+1] >> 8,
            img_out[3*(i*x_scaled_size+j)+2] >> 8);
        }
      }
    }
  }
  // fclose(f);
  return 0;
}
