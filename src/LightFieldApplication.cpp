/* 
 * File:   LightFieldApplication.cpp
 * Author: allan
 */

#include "LightFieldApplication.h"
#include "LightFieldImageLoader.h"
#include "LightFieldDrawable.h"

LightFieldApplication* LightFieldApplication::_instance = 0;

LightFieldApplication::LightFieldApplication() :
    _window( new MainWindow( "Light Field Rendering" ) )
{
    osg::ref_ptr< osgGA::TrackballManipulator > manipulator = new osgGA::TrackballManipulator();

    _scene = new osg::Group;
    _scene->setDataVariance( osg::Object::DYNAMIC );
    _scene->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF ); 
        
    _window->getCanvas().setCameraManipulator( manipulator );
    _window->getCanvas().getCamera()->setClearColor( osg::Vec4( .0f, .0f, .0f, 1.f ) );
    _window->getCanvas().setSceneData( _scene );
    _window->show();
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


bool LightFieldApplication::loadLightField( std::string lightFieldHeader )
{
    LightFieldImageLoader loader;
    
    if( !loader.load( lightFieldHeader ) )
        return false;
    
    //clear scene
    
    _lightFieldImage = loader.getLightFieldImage();
    
    _scene->addChild( createLightFieldNode() );
    
    return true;
}


osg::ref_ptr< osg::Group > LightFieldApplication::createLightFieldNode()
{
    LightFieldRender* lightFieldRender = new LightFieldRender( _lightFieldImage );    
    osg::ref_ptr< osg::Drawable > lightFieldDrawable = new LightFieldDrawable( lightFieldRender );
        
    osg::ref_ptr< osg::Geode > lightfieldGeode = new osg::Geode;
    lightfieldGeode->addDrawable( lightFieldDrawable );
    
    osg::ref_ptr< osg::Group > lightfieldGroup = new osg::Group;
    lightfieldGroup->addChild( lightfieldGeode );
    
    return lightfieldGroup;
}