/* 
 * File:   LightFieldRender.cpp
 * Author: allan
 */

#include "LightFieldRender.h"

LightFieldRender::LightFieldRender( LightFieldImage* lightFieldImage ) :
    _lightFieldImage( lightFieldImage )
{
}


LightFieldRender::~LightFieldRender() 
{
}


void LightFieldRender::render()
{
    
}


void LightFieldRender::getBoundingBox( float& xMin, float& xMax, float& yMin, float& yMax, float& zMin, float& zMax )
{
    xMin = xMax = yMin = yMax = zMin = zMax = 0;
}