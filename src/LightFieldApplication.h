/* 
 * File:   LightFieldApplication.h
 * Author: allan
 */

#ifndef LIGHTFIELDAPPLICATION_H
#define LIGHTFIELDAPPLICATION_H

#include "MainWindow.h"

#include <osg/Node>
#include <osg/Geode>

class LightFieldApplication 
{
public:
    
    virtual ~LightFieldApplication();
    
    static LightFieldApplication* getInstance();
    
private:
    
    LightFieldApplication();
    
    osg::ref_ptr< osg::Group > createLightFieldRender();

    
    static LightFieldApplication* _instance;
    
    MainWindow* _window;
    
    osg::ref_ptr< osg::Group > _scene;    
};

#endif /* LIGHTFIELDAPPLICATION_H */

