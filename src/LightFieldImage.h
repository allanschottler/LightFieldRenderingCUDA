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
    
    LightFieldImage( unsigned int nRows, unsigned int nCollumns );
    
    virtual ~LightFieldImage() {};
    
    void setMicroImage( unsigned int index1d, MicroImage microImage );
    
    MicroImage getMicroImage( unsigned int index1d );
    
    unsigned int getIndex1d( unsigned int row, unsigned int collumn );
    
    float* getTexels();
    
    void getDimensions( int& width, int& height );
    
private:

    unsigned int _nRows;
    unsigned int _nCollumns;
    
    std::vector< MicroImage > _microImages;
};

#endif	/* LIGHTFIELDIMAGE_H */

