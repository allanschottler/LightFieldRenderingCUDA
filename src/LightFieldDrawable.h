/* 
 * File:   LightFieldDrawable.h
 * Author: allan
 */

#ifndef LIGHTFIELDDRAWABLE_H
#define	LIGHTFIELDDRAWABLE_H

#include <osg/Drawable>

#include "LightFieldRender.h"

namespace LightField
{

class Drawable : public osg::Drawable
{
public:
    
    Drawable( Render* lightFieldRender );
    
    virtual ~Drawable();
    
    /**
    * Retorna a caixa envolvente
    */
    osg::BoundingBox computeBoundingBox() const;

    /**
    * Faz o desenho do objeto.
    */
    void drawImplementation( osg::RenderInfo& renderInfo ) const;
        
    /**
     * Construtor de cópia.
     */
    osg::Object* cloneType() const
    {
        return new Drawable( _lightFieldRender );
    }

    /**
     * Construtor de cópia.
     */
    osg::Object* clone( const osg::CopyOp& copyop ) const
    {
        return new Drawable( *this );
    }
    
private:

    Render* _lightFieldRender;
};

}

#endif	/* LIGHTFIELDDRAWABLE_H */

