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

namespace LightField
{
    
class Application 
{
public:
    
    virtual ~Application();
    
    static Application* getInstance();
    
    bool loadLightField( std::string lightFieldHeader );
    
    void createLightFieldNode();
    
    void setFocalPlane( float focalPlane );
    
private:
    
    Application();   
    
    
    static Application* _instance;
    
    MainWindow* _window;
    
    osg::ref_ptr< osg::Group > _scene;    
    
    ImageLoader _lightFieldLoader;
    
    Image* _lightFieldImage;
    
    Render* _lightFieldRender;
    
    float _focalPlane;
};

}

#endif /* LIGHTFIELDAPPLICATION_H */

