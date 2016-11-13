/* 
 * File:   LightFieldImageLoader.cpp
 * Author: allan
 */

#include "LightFieldImageLoader.h"
#include "FileName.h"
#include "ThreadManager.h"
#include "PNGLoaderThread.h"
#include "LightFieldApplication.h"

#include <sys/types.h>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

bool enumerateFiles( std::string dir, std::vector< std::string >& files )
{
    DIR *dp;
    struct dirent *dirp;
    
    if( ( dp  = opendir( dir.c_str() ) ) == NULL ) 
    {
        std::cout << "Error(" << errno << ") opening " << dir << std::endl;
        return false;
    }

    while( ( dirp = readdir( dp ) ) != NULL )
    {
        files.push_back( std::string( dirp->d_name ) );
    }
    
    closedir( dp );
    return true;
}


bool LightFieldImageLoader::load( std::string headerPath )
{
    _nFinishedThreads = _nFilesToLoad = 0;
    
    FileName::getDirectory( headerPath, _folderPath );
    FileName::getFullName( headerPath, _headerPath );
        
    if( !readHeader() )
        return false;
    
    std::vector< std::string > filePaths;
    
    if( enumerateFiles( _folderPath, filePaths ) )
    {   
        std::vector< std::string > microImagePaths;
        
        std::copy_if( filePaths.begin(), filePaths.end(), std::back_inserter( microImagePaths ), []( const std::string& path )
        {            
            std::string extension;
            FileName::getExtension( path, extension );
            
            return extension == ".png";
        });
        
        std::sort( microImagePaths.begin(), microImagePaths.end() );   
        
        _nFilesToLoad = microImagePaths.size();
        
        if( !_nFilesToLoad )
            return false;
        
        int threadID = 0;
        
        for( auto microImagePath : microImagePaths )
        {
            ThreadManager::getInstance()->startThread( this, 
                new PNGLoaderThread( _folderPath + microImagePath, threadID++ ) );
        }
        
        return true;
    }
    
    return false;
}
    
    
bool LightFieldImageLoader::readHeader()
{
    size_t nRows, nCollumns, imagesWidth, imagesHeight;
    
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
    
    if( tokens.size() != 4 )
        return false;
    
    nCollumns    = std::stoi( tokens[ 0 ], nullptr );
    nRows        = std::stoi( tokens[ 1 ], nullptr );
    imagesWidth  = std::stoi( tokens[ 2 ], nullptr );
    imagesHeight = std::stoi( tokens[ 3 ], nullptr );
    
    _lightFieldImage = new LightFieldImage( nRows, nCollumns, imagesWidth, imagesHeight );
    
    return true;
}
   

void LightFieldImageLoader::receiveThreadState( Thread* thread, const ThreadState& state )
{
    PNGLoaderThread* loaderThread = dynamic_cast< PNGLoaderThread* >( thread );
    
    switch( state )
    {
        case THREAD_CONCLUDED:
        {
            _nFinishedThreads++;
            
            _lightFieldImage->setMicroImage( loaderThread->getThreadID(), loaderThread->getLoadedImage() );
            
            if( _nFinishedThreads == _nFilesToLoad )
            {
                std::cout << "All done!\n";
                LightFieldApplication::getInstance()->createLightFieldNode();
            }
        }
            break;
        
        case THREAD_NOTIFICATION:
        case THREAD_ABORT:
        case THREAD_CANCELED:
            break;
}
}