/* 
 * File:   LightFieldImage.h
 * Author: allan
 */

#ifndef LIGHTFIELDIMAGE_H
#define	LIGHTFIELDIMAGE_H

#include <osg/Image>

class LightFieldImage 
{
public:
    
    typedef osg::ref_ptr< osg::Image > MicroImage;
    
    LightFieldImage( unsigned int nRows, unsigned int nCollumns );
    
    virtual ~LightFieldImage() {};
    
    void setMicroImage( unsigned int index1d, MicroImage microImage );
    
    MicroImage getMicroImage( unsigned int index1d );
    
    unsigned int getIndex1d( unsigned int row, unsigned int collumn );
    
private:

    unsigned int _nRows;
    unsigned int _nCollumns;
    
    std::vector< MicroImage > _microImages;
};

#endif	/* LIGHTFIELDIMAGE_H */

