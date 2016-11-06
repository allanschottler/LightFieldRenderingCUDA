/* 
 * File:   LightFieldImageLoader.cpp
 * Author: allan
 */

#include "LightFieldImageLoader.h"

#include <sys/types.h>
#include <dirent.h>
#include <osgDB/ReadFile>

int listdir( std::string dir, std::vector< std::string >& files );

bool LightFieldImageLoader::load( std::string headerPath )
{
    _headerPath = headerPath;
    
    std::string lastBar = "/";
    auto it = std::find_end( _headerPath.begin(), _headerPath.end(), lastBar.begin(), lastBar.end() );
    
    _folderPath = _headerPath.substr( 0, std::distance( _headerPath.begin(), it ) );
    _headerPath = _headerPath.substr( std::distance( _headerPath.begin(), it + 1 ), std::distance( it + 1, _headerPath.end() ) );
    
    if( !readHeader() )
        return false;
    
    std::vector< std::string > filePaths;
    
    if( !listdir( _folderPath, filePaths ) )
    {
        std::vector< std::string > microImagePaths;
        
        std::copy_if( filePaths.begin(), filePaths.end(), std::back_inserter( microImagePaths ), []( const std::string& path )
        {
            std::string pngFormat = "png";
            auto it = std::find_end( path.begin(), path.end(), pngFormat.begin(), pngFormat.end() );
            
            return it != path.end();
        });
        
        std::sort( microImagePaths.begin(), microImagePaths.end() );
        
        int index = 0;
        
        for( auto microImagePath : microImagePaths )
        {
            auto microImage = readMicroImage( microImagePath );
            
            std::cout << "LOADED " << microImagePath << "\n";
            _lightFieldImage->setMicroImage( index++, microImage );
        }
        
        return true;
    }
    
    return false;
}
    
    
bool LightFieldImageLoader::readHeader()
{
    unsigned int nRows, nCollumns;
    
    std::string line;
    std::ifstream myfile( _folderPath + "/" + _headerPath );
    
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
    return osgDB::readImageFile( _folderPath + "/" + imagePath ); 
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