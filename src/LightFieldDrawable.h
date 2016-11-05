/* 
 * File:   LightFieldDrawable.h
 * Author: allan
 */

#ifndef LIGHTFIELDDRAWABLE_H
#define	LIGHTFIELDDRAWABLE_H

#include <osg/Drawable>

#include "LightFieldRender.h"

class LightFieldDrawable : public osg::Drawable
{
public:
    
    LightFieldDrawable( LightFieldRender* lightFieldRender );
    
    virtual ~LightFieldDrawable();
    
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
        return new LightFieldDrawable( _lightFieldRender );
    }

    /**
     * Construtor de cópia.
     */
    osg::Object* clone( const osg::CopyOp& copyop ) const
    {
        return new LightFieldDrawable( *this );
    }
    
private:

    LightFieldRender* _lightFieldRender;
};

#endif	/* LIGHTFIELDDRAWABLE_H */

