/* 
 * File:   LightFieldImage.cpp
 * Author: allanws
 * 
 * Created on November 5, 2016, 6:41 PM
 */

#include <vector>

#include "LightFieldImage.h"

LightFieldImage::LightFieldImage( unsigned int nRows, unsigned int nCollumns ) :
    _nRows( nRows ),
    _nCollumns( nCollumns )
{
    _microImages.resize( _nRows * _nCollumns );
}


void LightFieldImage::setMicroImage( unsigned int index1d, LightFieldImage::MicroImage microImage )
{
    _microImages[ index1d ] = microImage;
}
    

LightFieldImage::MicroImage LightFieldImage::getMicroImage( unsigned int index1d )
{
    return _microImages[ index1d ];
}

    
unsigned int LightFieldImage::getIndex1d( unsigned int row, unsigned int collumn )
{
    return row * _nCollumns + collumn;
}
