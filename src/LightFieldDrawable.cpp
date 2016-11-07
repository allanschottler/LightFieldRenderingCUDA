/* 
 * File:   LightFieldDrawable.cpp
 * Author: allan
 */

#include "LightFieldDrawable.h"

LightFieldDrawable::LightFieldDrawable( LightFieldRender* lightFieldRender ) :
    _lightFieldRender( lightFieldRender )
{
    setUseDisplayList( false );
}


LightFieldDrawable::~LightFieldDrawable() 
{
}


osg::BoundingBox LightFieldDrawable::computeBoundingBox() const
{
    float xMin, xMax, yMin, yMax, zMin, zMax;
    _lightFieldRender->getBoundingBox( xMin, xMax, yMin, yMax, zMin, zMax );
    
    osg::Vec3 minimum( xMin, yMin, zMin ), maximum( xMax, yMax, zMax );
    return osg::BoundingBox( minimum, maximum );
}

   
void LightFieldDrawable::drawImplementation( osg::RenderInfo& renderInfo ) const
{
    _lightFieldRender->render();
}