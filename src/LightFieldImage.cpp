/* 
 * File:   LightFieldImage.cpp
 * Author: allanws
 * 
 * Created on November 5, 2016, 6:41 PM
 */

#include <vector>
#include <iostream>

#include "LightFieldImage.h"

namespace LightField
{

Image::Image( size_t nRows, size_t nCollumns, size_t microImageWidth, size_t microImageHeight ) :
    _nRows( nRows ),
    _nCollumns( nCollumns ),
    _microImageWidth( microImageWidth ),
    _microImageHeight( microImageHeight )
{
    _microImages.resize( _nRows * _nCollumns );
    _texels = new unsigned char[ _nRows * _nCollumns * _microImageWidth * _microImageHeight * 4 ];
}


void Image::setMicroImage( size_t index1d, Image::MicroImage microImage )
{
    _microImages[ index1d ] = microImage;
    
    size_t iCollumn = index1d % _nCollumns;
    size_t iRow = ( index1d - iCollumn ) / _nCollumns;
    
    for( size_t y = 0; y < _microImageHeight; y++ ) 
    {
        for( size_t x = 0; x < _microImageWidth; x++ ) 
        {            
            size_t i = y * _microImageWidth + x;
            size_t t = iRow * _nCollumns * _microImageWidth * _microImageHeight +
                    y * _nCollumns * _microImageWidth +
                    iCollumn * _microImageWidth + x;

            _texels[ 4 * t ]     = microImage[ 4 * i ]; /// 255.;
            _texels[ 4 * t + 1 ] = microImage[ 4 * i + 1 ]; /// 255.;
            _texels[ 4 * t + 2 ] = microImage[ 4 * i + 2 ]; /// 255.;
            _texels[ 4 * t + 3 ] = microImage[ 4 * i + 3 ]; /// 255.;
        }
    }
}
    

Image::MicroImage Image::getMicroImage( size_t index1d )
{
    return _microImages[ index1d ];
}

    
size_t Image::getIndex1d( size_t row, size_t collumn )
{
    return row * _nCollumns + collumn;
}


unsigned char* Image::getTexels()
{
    return _texels;
}


void Image::getTextureDimensions( int& width, int& height )
{
    width = _microImageWidth;
    height = _microImageHeight;
}


void Image::getDimensions( int& width, int& height )
{
    width = _nCollumns;
    height = _nRows;
}

}