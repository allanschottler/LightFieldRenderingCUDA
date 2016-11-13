/* 
 * File:   LightFieldImage.h
 * Author: allan
 */

#ifndef LIGHTFIELDIMAGE_H
#define	LIGHTFIELDIMAGE_H

#include <vector>

class LightFieldImage 
{
public:
    
    typedef std::vector< unsigned char > MicroImage;
    
    LightFieldImage( size_t nRows, size_t nCollumns, size_t microImageWidth, size_t microImageHeight );
    
    virtual ~LightFieldImage() {};
    
    void setMicroImage( size_t index1d, MicroImage microImage );
    
    MicroImage getMicroImage( size_t index1d );
    
    size_t getIndex1d( size_t row, size_t collumn );
    
    unsigned char* getTexels();
        
    void getTextureDimensions( int& width, int& height );
    
    void getDimensions( int& width, int& height );
    
private:

    size_t _nRows;
    size_t _nCollumns;
    
    size_t _microImageWidth;
    size_t _microImageHeight;
    
    std::vector< MicroImage > _microImages;
};

#endif	/* LIGHTFIELDIMAGE_H */

