/* 
 * File:   LightFieldImageLoader.h
 * Author: allan
 */

#ifndef LIGHTFIELDIMAGELOADER_H
#define	LIGHTFIELDIMAGELOADER_H

#include "LightFieldImage.h"

#include <string>

class LightFieldImageLoader 
{
public:
    
    LightFieldImageLoader() : _lightFieldImage( nullptr ) {};
    
    virtual ~LightFieldImageLoader() {};
    
    bool load( std::string headerPath );
    
    LightFieldImage* getLightFieldImage() { return _lightFieldImage; };
    
private:
    
    bool readHeader();
    
    LightFieldImage::MicroImage readMicroImage( std::string imagePath );
    
    std::string _folderPath;
    
    std::string _headerPath;
    
    LightFieldImage* _lightFieldImage;

};

#endif	/* LIGHTFIELDIMAGELOADER_H */

