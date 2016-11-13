/* 
 * File:   PNGLoaderThread.cpp
 * Author: allan
 */

#include "PNGLoaderThread.h"

#include <png.h>
#include <iostream>

bool LoadPNG24( unsigned char ** pixelBuffer, const char *filename, unsigned int *width, unsigned int *height ) 
{
    png_byte header[8];

    //open file as binary 
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return false;
    }

    //read the header
    fread(header, 1, 8, fp);

    //test if png
    int is_png = !png_sig_cmp(header, 0, 8);
    if (!is_png) {
        fclose(fp);
        return false;
    }

    //create png struct
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return (false);
    }

    //create png info struct
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, (png_infopp) NULL, (png_infopp) NULL);
        fclose(fp);
        return (false);
    }

    //create png info struct
    png_infop end_info = png_create_info_struct(png_ptr);
    if (!end_info) {
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) NULL);
        fclose(fp);
        return (false);
    }

    //png error stuff, not sure libpng man suggests this.
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        fclose(fp);
        return (false);
    }

    //init png reading
    png_init_io(png_ptr, fp);

    //let libpng know you already read the first 8 bytes
    png_set_sig_bytes(png_ptr, 8);

    // read all the info up to the image data
    png_read_info(png_ptr, info_ptr);

    //variables to pass to get info
    int bit_depth, color_type;
    png_uint_32 twidth, theight;

    // get info about png
    png_get_IHDR(png_ptr, info_ptr, &twidth, &theight, &bit_depth, &color_type, NULL, NULL, NULL);

    // Update the png info struct.
    //png_read_update_info(png_ptr, info_ptr);

    // Row size in bytes.
    //png_size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    png_size_t rowbytes = sizeof(png_byte) * 4 * twidth;

    // Allocate the image_data as a big block, to be given to opencl
    png_byte *image_data = (png_byte *)malloc(sizeof(png_byte) * 4 * twidth * theight);
    if (!image_data) {
        //clean up memory and close stuff
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        fclose(fp);
        return false;
    }

    //row_pointers is for pointing to image_data for reading the png with libpng
    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * theight);
    if (!row_pointers) {
        //clean up memory and close stuff
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        free(image_data);
        fclose(fp);
        return false;
    }

    // set the individual row_pointers to point at the correct offsets of image_data
    for (unsigned int i = 0; i < theight; ++i) {
        row_pointers[i] = image_data + (i * rowbytes);
    }

    // PNG Transforms
    if (color_type == PNG_COLOR_TYPE_RGB) {
        //png_set_filler(png_ptr, 0xff, PNG_FILLER_AFTER);
        png_set_add_alpha(png_ptr, 0xff, PNG_FILLER_AFTER);
    }

    png_read_update_info(png_ptr, info_ptr);
    
    //read the png into image_data through row_pointers
    png_read_image(png_ptr, row_pointers);

    //clean up memory and close stuff
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info); 
    free(row_pointers);
    fclose(fp);

    //update width and height based on png info
    (*pixelBuffer) = image_data;
    if (width) {*width = twidth;}
    if (height) {*height = theight;}

    return true;
}


PNGLoaderThread::PNGLoaderThread( std::string pngFilePath, int threadID ) :
    _pngFilePath( pngFilePath ),
    _threadID( threadID )
{
}


PNGLoaderThread::~PNGLoaderThread() 
{
    cancelCleanUp();
}


void PNGLoaderThread::executeStep()
{
    unsigned char* imageBuffer;
    unsigned int width, height;
    
    if( LoadPNG24( &imageBuffer, ( _pngFilePath ).c_str(), &width, &height ) )
    {
        _loadedImage.assign( imageBuffer, imageBuffer + width * height * 4 );
        free( imageBuffer );
        
        std::cout << "LOADED " << _pngFilePath << "\n";
    }
    
    finish();
}


void PNGLoaderThread::cancelCleanUp()
{
    // TODO
}