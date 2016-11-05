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

class LightFieldApplication 
{
public:
    
    virtual ~LightFieldApplication();
    
    static LightFieldApplication* getInstance();
    
    void loadLightField( std::string lightFieldHeader );
    
private:
    
    LightFieldApplication();   
    
    osg::ref_ptr< osg::Group > createLightFieldNode();

    
    static LightFieldApplication* _instance;
    
    MainWindow* _window;
    
    osg::ref_ptr< osg::Group > _scene;    
    
    LightFieldImage* _lightFieldImage;
};

#endif /* LIGHTFIELDAPPLICATION_H */

