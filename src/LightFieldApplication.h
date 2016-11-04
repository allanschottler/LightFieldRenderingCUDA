/* 
 * File:   LightFieldApplication.h
 * Author: allan
 *
 * Created on April 12, 2016, 11:48 PM
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
    
    
    static LightFieldApplication* _instance;
    
    MainWindow* _window;
    
    osg::ref_ptr< osg::Group > _scene;    
};

#endif /* LIGHTFIELDAPPLICATION_H */

