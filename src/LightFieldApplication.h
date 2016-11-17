/* 
 * File:   LightFieldApplication.h
 * Author: allan
 */

#ifndef LIGHTFIELDAPPLICATION_H
#define LIGHTFIELDAPPLICATION_H

#include "MainWindow.h"

#include <osg/Node>
#include <osg/Geode>

#include "LightFieldImage.h"
#include "LightFieldImageLoader.h"
#include "LightFieldRender.h"

class LightFieldApplication 
{
public:
    
    virtual ~LightFieldApplication();
    
    static LightFieldApplication* getInstance();
    
    bool loadLightField( std::string lightFieldHeader );
    
    void createLightFieldNode();
    
    void setFocalPlane( float focalPlane );
    
private:
    
    LightFieldApplication();   
    
    
    static LightFieldApplication* _instance;
    
    MainWindow* _window;
    
    osg::ref_ptr< osg::Group > _scene;    
    
    LightFieldImageLoader _lightFieldLoader;
    
    LightFieldImage* _lightFieldImage;
    
    LightFieldRender* _lightFieldRender;
    
    float _focalPlane;
};

#endif /* LIGHTFIELDAPPLICATION_H */

