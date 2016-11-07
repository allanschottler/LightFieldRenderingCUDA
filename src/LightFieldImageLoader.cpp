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

int listdir( std::string dir, std::vector< std::string >& files );
bool loadPngImage( const char *name, int &outWidth, int &outHeight, bool &outHasAlpha, unsigned char **outData );

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
    
    _lightFieldImage = new LightFieldImage( nRows, nCollumns );  
    
    return true;
}
    

LightFieldImage::MicroImage LightFieldImageLoader::readMicroImage( std::string imagePath )
{
    LightFieldImage::MicroImage microImage;
    unsigned char* readImage;
    int width, height;
    bool hasAlpha;
    
    if( loadPngImage( (_folderPath + imagePath).c_str(), width, height, hasAlpha, &readImage ) )
    {
        microImage.assign( readImage, readImage + width * height );
        free( readImage );
    }
        
    return microImage;
}


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

bool loadPngImage( const char *name, int &outWidth, int &outHeight, bool &outHasAlpha, unsigned char **outData ) 
{
    png_structp png_ptr;
    png_infop info_ptr;
    unsigned int sig_read = 0;
    int color_type, interlace_type;
    FILE *fp;

    if ((fp = fopen(name, "rb")) == NULL)
        return false;

    /* Create and initialize the png_struct
     * with the desired error handler
     * functions.  If you want to use the
     * default stderr and longjump method,
     * you can supply NULL for the last
     * three parameters.  We also supply the
     * the compiler header file version, so
     * that we know if the application
     * was compiled with a compatible version
     * of the library.  REQUIRED
     */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
            NULL, NULL, NULL);

    if (png_ptr == NULL) {
        fclose(fp);
        return false;
    }

    /* Allocate/initialize the memory
     * for image information.  REQUIRED. */
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
        fclose(fp);
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return false;
    }

    /* Set error handling if you are
     * using the setjmp/longjmp method
     * (this is the normal method of
     * doing things with libpng).
     * REQUIRED unless you  set up
     * your own error handlers in
     * the png_create_read_struct()
     * earlier.
     */
    if (setjmp(png_jmpbuf(png_ptr))) {
        /* Free all of the memory associated
         * with the png_ptr and info_ptr */
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        /* If we get here, we had a
         * problem reading the file */
        return false;
    }

    /* Set up the output control if
     * you are using standard C streams */
    png_init_io(png_ptr, fp);

    /* If we have already
     * read some of the signature */
    png_set_sig_bytes(png_ptr, sig_read);

    /*
     * If you have enough memory to read
     * in the entire image at once, and
     * you need to specify only
     * transforms that can be controlled
     * with one of the PNG_TRANSFORM_*
     * bits (this presently excludes
     * dithering, filling, setting
     * background, and doing gamma
     * adjustment), then you can read the
     * entire image (including pixels)
     * into the info structure with this
     * call
     *
     * PNG_TRANSFORM_STRIP_16 |
     * PNG_TRANSFORM_PACKING  forces 8 bit
     * PNG_TRANSFORM_EXPAND forces to
     *  expand a palette into RGB
     */
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND, NULL);

    png_uint_32 width, height;
    int bit_depth;
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
            &interlace_type, NULL, NULL);
    outWidth = width;
    outHeight = height;

    unsigned int row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    *outData = (unsigned char*) malloc(row_bytes * outHeight);

    png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);

    for (int i = 0; i < outHeight; i++) {
        // note that png is ordered top to
        // bottom, but OpenGL expect it bottom to top
        // so the order or swapped
        memcpy(*outData + (row_bytes * (outHeight - 1 - i)), row_pointers[i], row_bytes);
    }

    /* Clean up after the read,
     * and free any memory allocated */
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

    /* Close the file */
    fclose(fp);

    /* That's it */
    return true;
}