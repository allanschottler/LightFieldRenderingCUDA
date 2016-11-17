/* 
 * File:   LightFieldDrawable.cpp
 * Author: allan
 */

#include "LightFieldDrawable.h"

namespace LightField
{

Drawable::Drawable( Render* lightFieldRender ) :
    _lightFieldRender( lightFieldRender )
{
    setUseDisplayList( false );
}


Drawable::~Drawable() 
{
}


osg::BoundingBox Drawable::computeBoundingBox() const
{
    float xMin, xMax, yMin, yMax, zMin, zMax;
    _lightFieldRender->getBoundingBox( xMin, xMax, yMin, yMax, zMin, zMax );
    
    osg::Vec3 minimum( xMin, yMin, zMin ), maximum( xMax, yMax, zMax );
    return osg::BoundingBox( minimum, maximum );
}

   
void Drawable::drawImplementation( osg::RenderInfo& renderInfo ) const
{
    _lightFieldRender->render();
}

}