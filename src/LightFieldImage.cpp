/* 
 * File:   LightFieldImage.cpp
 * Author: allanws
 * 
 * Created on November 5, 2016, 6:41 PM
 */

#include <vector>
#include <iostream>

#include "LightFieldImage.h"

LightFieldImage::LightFieldImage( size_t nRows, size_t nCollumns, size_t microImageWidth, size_t microImageHeight ) :
    _nRows( nRows ),
    _nCollumns( nCollumns ),
    _microImageWidth( microImageWidth ),
    _microImageHeight( microImageHeight )
{
    _microImages.resize( _nRows * _nCollumns );
}


void LightFieldImage::setMicroImage( size_t index1d, LightFieldImage::MicroImage microImage )
{
    _microImages[ index1d ] = microImage;
}
    

LightFieldImage::MicroImage LightFieldImage::getMicroImage( size_t index1d )
{
    return _microImages[ index1d ];
}

    
size_t LightFieldImage::getIndex1d( size_t row, size_t collumn )
{
    return row * _nCollumns + collumn;
}


unsigned char* LightFieldImage::getTexels()
{
    //unsigned char* texels = new unsigned char[ _nRows * _nCollumns * _microImageWidth * _microImageHeight * 4 ];//[ _microImageWidth * _microImageHeight * 4 ]; //
    unsigned char* texels = new unsigned char[ _microImageWidth * _microImageHeight * 4 ]; //
    LightFieldImage::MicroImage microImage = getMicroImage( 1 );
    
    for( size_t i = 0; i < _microImageWidth * _microImageHeight * 4; i += 4 )
    {        
        //size_t microImageIndex = i / _microImageWidth
        texels[ i ]     = microImage[ i ];
        texels[ i + 1 ] = microImage[ i + 1 ];
        texels[ i + 2 ] = microImage[ i + 2 ];
        texels[ i + 3 ] = microImage[ i + 3 ];
    }
    
    /*for( size_t i = 0; i < _nRows * _nCollumns * _microImageWidth * _microImageHeight * 4; i += 4 )
    {
        size_t x = i % (_nCollumns * _microImageHeight * 4);
        size_t y = 0;
        
        //size_t microImageIndex = i / _microImageWidth
        texels[ i ]     = 1.0;
        texels[ i + 1 ] = (i % (_microImageWidth / 4) == 0) ? 1.0 : 0.0;
        texels[ i + 2 ] = 1.0;
        texels[ i + 3 ] = 1.0;
    }*/
    /*for( size_t iRow = 0; iRow < _nRows; iRow++ )    
    {
        for( size_t iCollumn = 0; iCollumn < _nCollumns; iCollumn++ )
        {
            size_t microImageIndex = getIndex1d( iRow, iCollumn );
            LightFieldImage::MicroImage microImage = getMicroImage( microImageIndex );
            
            for( size_t y = 0; y < _microImageHeight; y++ )
            {
                for( size_t x = 0; x < _microImageWidth; x++ )
                {
                    size_t i = y * _microImageWidth + x;
                    size_t t = iRow * _nCollumns * _microImageWidth * _microImageHeight + 
                               y * _nCollumns * _microImageWidth + 
                               iCollumn * _microImageWidth + x;
                    
                    texels[ t ]     = microImage[ i ]     ;/// 255.;
                    texels[ t + 1 ] = microImage[ i + 1 ] ;/// 255.;
                    texels[ t + 2 ] = microImage[ i + 2 ] ;/// 255.;
                    texels[ t + 3 ] = microImage[ i + 3 ] ;/// 255.;
                }
            }
        }
    }*/
    
    return texels;
}


void LightFieldImage::getTextureDimensions( int& width, int& height )
{
    width = _nCollumns * _microImageWidth;
    height = _nRows * _microImageHeight;
}


void LightFieldImage::getDimensions( int& width, int& height )
{
    width = _nCollumns;
    height = _nRows;
}