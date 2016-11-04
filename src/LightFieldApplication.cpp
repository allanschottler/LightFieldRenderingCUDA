/* 
 * File:   LightFieldApplication.cpp
 * Author: allan
 * 
 * Created on April 12, 2016, 11:48 PM
 */

#include "LightFieldApplication.h"

LightFieldApplication* LightFieldApplication::_instance = 0;

LightFieldApplication::LightFieldApplication() :
    _window( new MainWindow( "Light Field Rendering" ) )
{
}


LightFieldApplication::~LightFieldApplication()
{
    delete _instance;    
    _instance = 0;
}


LightFieldApplication* LightFieldApplication::getInstance()
{
    if( !_instance )
        _instance = new LightFieldApplication();
    
    return _instance;
}
