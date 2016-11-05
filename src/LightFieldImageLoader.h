/* 
 * File:   LightFieldImageLoader.h
 * Author: allan
 */

#ifndef LIGHTFIELDIMAGELOADER_H
#define	LIGHTFIELDIMAGELOADER_H

#include "LightFieldImage.h"

class LightFieldImageLoader 
{
public:
    
    LightFieldImageLoader() {};
    
    virtual ~LightFieldImageLoader() {};
    
    LightFieldImage* load( std::string headerPath );
    
private:
    
    void readHeader();
    
    LightFieldImage::MicroImage readMicroImage( std::string imagePath );
    
    std::string _headerPath;
    
    LightFieldImage* _lightFieldImage;

};

#endif	/* LIGHTFIELDIMAGELOADER_H */

