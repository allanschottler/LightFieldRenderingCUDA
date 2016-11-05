/* 
 * File:   LightFieldImageLoader.cpp
 * Author: allan
 */

#include "LightFieldImageLoader.h"

#include <sys/types.h>
#include <dirent.h>
#include <osgDB/ReadFile>

int listdir( std::string dir, std::vector< std::string >& files );

LightFieldImage* LightFieldImageLoader::load( std::string headerPath )
{
    _headerPath = headerPath;
    
    readHeader();
    
    std::vector< std::string > microImagePaths;
    
    if( !listdir( _headerPath, microImagePaths ) )
    {
        int index = 0;
        
        for( auto microImagePath : microImagePaths )
        {
            auto microImage = readMicroImage( microImagePath );
            _lightFieldImage->setMicroImage( index++, microImage );
        }
    }
    
    return _lightFieldImage;
}
    
    
void LightFieldImageLoader::readHeader()
{
    unsigned int nRows, nCollumns;
    
    // read header
    
    _lightFieldImage = new LightFieldImage( 0, 0 );
    
    std::string lastBar = "/";
    auto it = std::find_end( _headerPath.begin(), _headerPath.end(), lastBar.begin(), lastBar.end() );
    
    _headerPath = _headerPath.substr( 0, std::distance( _headerPath.begin(), it ) );
}
    

LightFieldImage::MicroImage LightFieldImageLoader::readMicroImage( std::string imagePath )
{
    return osgDB::readImageFile( imagePath ); 
}


int listdir( std::string dir, std::vector< std::string >& files )
{
    DIR *dp;
    struct dirent *dirp;
    
    if( (dp  = opendir( dir.c_str() ) ) == NULL ) 
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