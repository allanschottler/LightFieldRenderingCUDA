/* 
 * File:   LightFieldImageLoader.cpp
 * Author: allan
 */

#include "LightFieldImageLoader.h"
#include "FileName.h"

#include <sys/types.h>
#include <dirent.h>
#include <png.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

int listdir( std::string dir, std::vector< std::string >& files )
{
    DIR *dp;
    struct dirent *dirp;
    
    if( ( dp  = opendir( dir.c_str() ) ) == NULL ) 
    {
        std::cout << "Error(" << errno << ") opening " << dir << std::endl;
        return errno;
    }

    while( ( dirp = readdir( dp ) ) != NULL )
    {
        files.push_back( std::string( dirp->d_name ) );
    }
    
    closedir( dp );
    return 0;
}


bool LoadPNG24(unsigned char ** pixelBuffer, const char *filename, unsigned int *width, unsigned int *height) {
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

bool LightFieldImageLoader::load( std::string headerPath )
{
    FileName::getDirectory( headerPath, _folderPath );
    FileName::getFullName( headerPath, _headerPath );
        
    if( !readHeader() )
        return false;
    
    std::vector< std::string > filePaths;
    
    if( !listdir( _folderPath, filePaths ) )
    {
        std::vector< std::string > microImagePaths;
        
        std::copy_if( filePaths.begin(), filePaths.end(), std::back_inserter( microImagePaths ), []( const std::string& path )
        {            
            std::string extension;
            FileName::getExtension( path, extension );
            
            return extension == ".png";
        });
        
        std::sort( microImagePaths.begin(), microImagePaths.end() );
        
        int index = 0;
        
        for( auto microImagePath : microImagePaths )
        {
            auto microImage = readMicroImage( microImagePath );                 
            
            _lightFieldImage->setMicroImage( index++, microImage );  
            
            std::cout << "LOADED " << microImagePath << "\n";
        }
        
        return true;
    }
    
    return false;
}
    
    
bool LightFieldImageLoader::readHeader()
{
    unsigned int nRows, nCollumns;
    
    std::string line;
    std::ifstream myfile( _folderPath + _headerPath );
    
    if( myfile.is_open() )
    {
        getline( myfile, line );
        myfile.close();
    }
    else 
    {
        std::cout << "Unable to open file";
        return false;
    }
    
    std::string buf;
    std::stringstream ss( line );

    std::vector< std::string > tokens; 

    while( ss >> buf )
        tokens.push_back( buf );
    
    if( tokens.size() != 2 )
        return false;
    
    nRows = std::stoi( tokens[ 0 ], nullptr );
    nCollumns = std::stoi( tokens[ 1 ], nullptr );
    
    _lightFieldImage = new LightFieldImage( nRows, nCollumns, 1024, 1024 );  // HARDCOOOOODE
    
    return true;
}
    

LightFieldImage::MicroImage LightFieldImageLoader::readMicroImage( std::string imagePath )
{
    LightFieldImage::MicroImage microImage;
    unsigned char* imageBuffer;
    unsigned int width, height;
    
    if( LoadPNG24( &imageBuffer, (_folderPath + imagePath).c_str(), &width, &height ) )
    //if( loadPngImage( (_folderPath + imagePath).c_str(), width, height, hasAlpha, &readImage ) )
    {
        microImage.assign( imageBuffer, imageBuffer + width * height * 4 );
        free( imageBuffer );
    }
        
    return microImage;
}

//bool loadPngImage( const char *name, int &outWidth, int &outHeight, bool &outHasAlpha, unsigned char **outData ) 
//{
//    png_structp png_ptr;
//    png_infop info_ptr;
//    unsigned int sig_read = 0;
//    int color_type, interlace_type;
//    FILE *fp;
// 
//    if ((fp = fopen(name, "rb")) == NULL)
//        return false;
// 
//    /* Create and initialize the png_struct
//     * with the desired error handler
//     * functions.  If you want to use the
//     * default stderr and longjump method,
//     * you can supply NULL for the last
//     * three parameters.  We also supply the
//     * the compiler header file version, so
//     * that we know if the application
//     * was compiled with a compatible version
//     * of the library.  REQUIRED
//     */
//    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
//                                     NULL, NULL, NULL);
// 
//    if (png_ptr == NULL) {
//        fclose(fp);
//        return false;
//    }
// 
//    /* Allocate/initialize the memory
//     * for image information.  REQUIRED. */
//    info_ptr = png_create_info_struct(png_ptr);
//    if (info_ptr == NULL) {
//        fclose(fp);
//        png_destroy_read_struct(&png_ptr, NULL, NULL);
//        return false;
//    }
// 
//    /* Set error handling if you are
//     * using the setjmp/longjmp method
//     * (this is the normal method of
//     * doing things with libpng).
//     * REQUIRED unless you  set up
//     * your own error handlers in
//     * the png_create_read_struct()
//     * earlier.
//     */
//    if (setjmp(png_jmpbuf(png_ptr))) {
//        /* Free all of the memory associated
//         * with the png_ptr and info_ptr */
//        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
//        fclose(fp);
//        /* If we get here, we had a
//         * problem reading the file */
//        return false;
//    }
// 
//    /* Set up the output control if
//     * you are using standard C streams */
//    png_init_io(png_ptr, fp);
// 
//    /* If we have already
//     * read some of the signature */
//    png_set_sig_bytes(png_ptr, sig_read);
// 
//    /*
//     * If you have enough memory to read
//     * in the entire image at once, and
//     * you need to specify only
//     * transforms that can be controlled
//     * with one of the PNG_TRANSFORM_*
//     * bits (this presently excludes
//     * dithering, filling, setting
//     * background, and doing gamma
//     * adjustment), then you can read the
//     * entire image (including pixels)
//     * into the info structure with this
//     * call
//     *
//     * PNG_TRANSFORM_STRIP_16 |
//     * PNG_TRANSFORM_PACKING  forces 8 bit
//     * PNG_TRANSFORM_EXPAND forces to
//     *  expand a palette into RGB
//     */
//    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND, NULL);
// 
//    png_uint_32 width, height;
//    int bit_depth;
//    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
//                 &interlace_type, NULL, NULL);
//    outWidth = width;
//    outHeight = height;
//    
//    // PNG Transforms
//    if (color_type == PNG_COLOR_TYPE_RGB) {
//        //png_set_filler(png_ptr, 0xff, PNG_FILLER_AFTER);
//        png_set_add_alpha(png_ptr, 0xff, PNG_FILLER_AFTER);
//    }
// 
//    png_read_update_info(png_ptr, info_ptr);
//    
//    unsigned int row_bytes = png_get_rowbytes(png_ptr, info_ptr);
//    *outData = (unsigned char*) malloc(row_bytes * outHeight);
// 
//    png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);
// 
//    for (int i = 0; i < outHeight; i++) {
//        // note that png is ordered top to
//        // bottom, but OpenGL expect it bottom to top
//        // so the order or swapped
//        memcpy(*outData+(row_bytes * (outHeight-1-i)), row_pointers[i], row_bytes);
//    }
// 
//    /* Clean up after the read,
//     * and free any memory allocated */
//    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
// 
//    /* Close the file */
//    fclose(fp);
// 
//    /* That's it */
//    return true;
//}