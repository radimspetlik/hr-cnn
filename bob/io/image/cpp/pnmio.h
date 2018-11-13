/*
 * File       : pnmio.h
 * Description: Header file for pnmio.cpp.
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

#ifndef PNMIO_H
#define PNMIO_H

#include <stdio.h>

/* PNM/PFM image data file format definitions. */
#define PBM_ASCII         1
#define PGM_ASCII         2
#define PPM_ASCII         3
#define PBM_BINARY        4
#define PGM_BINARY        5
#define PPM_BINARY        6
#define PAM               7 /* reserved */
                            /* 8-15: reserved */
#define PFM_RGB          16 /* F */
#define PFM_GREYSCALE    17 /* f */

#define IS_BIGENDIAN(x)   ((*(char*)&x) == 0)
#define IS_LITTLE_ENDIAN  (1 == *(unsigned char *)&(const int){1})
#ifndef FALSE
#define FALSE             0
#endif
#ifndef TRUE
#define TRUE              1
#endif


/* PNM/PFM API. */

extern "C"{
int  get_pnm_type(FILE *f);
int read_pbm_header(FILE *f, int *img_xdim, int *img_ydim, int *is_ascii);
int read_pgm_header(FILE *f, int *img_xdim, int *img_ydim, int *img_colors,
       int *is_ascii);
int read_ppm_header(FILE *f, int *img_xdim, int *img_ydim, int *img_colors,
       int *is_ascii);
int read_pbm_data(FILE *f, int *img_in, int img_size, int is_ascii, int img_width);
int read_pgm_data(FILE *f, int *img_in, int img_size, int is_ascii, unsigned int bytes_per_sample);
int read_ppm_data(FILE *f, int *img_in, int img_size, int is_ascii, unsigned int bytes_per_sample);
int write_pbm_file(FILE *f, int *img_out,
       int x_size, int y_size, int x_scale_val, int y_scale_val, int linevals,
       int is_ascii);
int write_pgm_file(FILE *f, int *img_out,
       int x_size, int y_size, int x_scale_val, int y_scale_val,
       int img_colors, int linevals, int is_ascii, unsigned int bytes_per_sample);
int write_ppm_file(FILE *f, int *img_out,
       int x_size, int y_size, int x_scale_val, int y_scale_val,
       int img_colors, int is_ascii, unsigned int bytes_per_sample);
}

#endif /* PNMIO_H */
