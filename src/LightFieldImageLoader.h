/* 
 * File:   LightFieldImageLoader.h
 * Author: allan
 */

#ifndef LIGHTFIELDIMAGELOADER_H
#define	LIGHTFIELDIMAGELOADER_H

#include "LightFieldImage.h"
#include "ThreadListener.h"

#include <string>

namespace LightField
{

class ImageLoader : public ThreadListener
{
public:
    
    ImageLoader() : _lightFieldImage( nullptr ) {};
    
    virtual ~ImageLoader() {};
    
    bool load( std::string headerPath );    

    void receiveThreadState( Thread* thread, const ThreadState& state );

    Image* getLightFieldImage() { return _lightFieldImage; };
    
private:
    
    bool readHeader();
    
    //LightFieldImage::MicroImage readMicroImage( std::string imagePath );
    
    std::string _folderPath;
    
    std::string _headerPath;
    
    Image* _lightFieldImage;
    
    int _nFilesToLoad, _nFinishedThreads;

};

}

#endif	/* LIGHTFIELDIMAGELOADER_H */

