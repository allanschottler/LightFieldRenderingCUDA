/* 
 * File:   LightFieldImageLoader.h
 * Author: allan
 */

#ifndef LIGHTFIELDIMAGELOADER_H
#define	LIGHTFIELDIMAGELOADER_H

#include "LightFieldImage.h"
#include "ThreadListener.h"

#include <string>

class LightFieldImageLoader : public ThreadListener
{
public:
    
    LightFieldImageLoader() : _lightFieldImage( nullptr ) {};
    
    virtual ~LightFieldImageLoader() {};
    
    bool load( std::string headerPath );    

    void receiveThreadState( Thread* thread, const ThreadState& state );

    LightFieldImage* getLightFieldImage() { return _lightFieldImage; };
    
private:
    
    bool readHeader();
    
    //LightFieldImage::MicroImage readMicroImage( std::string imagePath );
    
    std::string _folderPath;
    
    std::string _headerPath;
    
    LightFieldImage* _lightFieldImage;
    
    int _nFilesToLoad, _nFinishedThreads;

};

#endif	/* LIGHTFIELDIMAGELOADER_H */

