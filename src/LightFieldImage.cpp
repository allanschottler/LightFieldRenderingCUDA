/* 
 * File:   LightFieldImage.cpp
 * Author: allanws
 * 
 * Created on November 5, 2016, 6:41 PM
 */

#include <vector>

#include "LightFieldImage.h"

LightFieldImage::LightFieldImage( unsigned int nRows, unsigned int nCollumns, unsigned int microImageWidth, unsigned int microImageHeight ) :
    _nRows( nRows ),
    _nCollumns( nCollumns ),
    _microImageWidth( microImageWidth ),
    _microImageHeight( microImageHeight )
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


float* LightFieldImage::getTexels()
{
    float* texels = new float[ _microImageWidth * _microImageHeight * 4 ]; //[ _nRows * _nCollumns * _microImageWidth * _microImageHeight * 4 ];
    
    for( size_t i = 0; i < _microImageWidth * _microImageHeight * 4; i += 4 )
    {
        texels[ i ]     = 1.0;
        texels[ i + 1 ] = (i % (_microImageWidth / 4) == 0) ? 1.0 : 0.0;
        texels[ i + 2 ] = 1.0;
        texels[ i + 3 ] = 1.0;
    }
    /*for( unsigned int iRow = 0; iRow < _nRows; iRow++ )    
    {
        for( unsigned int iCollumn = 0; iCollumn < _nCollumns; iCollumn++ )
        {
            unsigned int index = getIndex1d( iRow, iCollumn );
            LightFieldImage::MicroImage microImage = getMicroImage( index );
            
        }
    }*/
    
    return texels;
}


void LightFieldImage::getDimensions( int& width, int& height )
{
    width = _nRows;
    height = _nCollumns;
}


void LightFieldImage::getMicroImageDimensions( int& width, int& height )
{
    width = _microImageWidth;
    height = _microImageHeight;
}