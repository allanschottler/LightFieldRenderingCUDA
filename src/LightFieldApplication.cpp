/* 
 * File:   LightFieldApplication.cpp
 * Author: allan
 */

#include "LightFieldApplication.h"

LightFieldApplication* LightFieldApplication::_instance = 0;

LightFieldApplication::LightFieldApplication() :
    _window( new MainWindow( "Light Field Rendering" ) )
{
    osg::ref_ptr< osgGA::TrackballManipulator > manipulator = new osgGA::TrackballManipulator();

    _scene = new osg::Group;
    _scene->setDataVariance( osg::Object::DYNAMIC );
    _scene->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF ); 
    
    // Build scene
    _scene->addChild( createLightFieldRender() );
    
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


osg::ref_ptr< osg::Group > LightFieldApplication::createLightFieldRender()
{
    osg::ref_ptr< osg::Group > lightfieldGroup = new osg::Group;
    osg::ref_ptr< osg::Geode > lightfieldGeode = new osg::Geode;
    osg::ref_ptr< osg::Geometry > lightfieldGeometry = new osg::Geometry;
    
//    osg::ref_ptr< osg::Vec3Array > vertexArray = new osg::Vec3Array;
//    osg::ref_ptr< osg::Vec4Array > colorArray = new osg::Vec4Array;
    //osg::ref_ptr< osg::Vec3Array > vertexArray = new osg::Vec3Array;
    
//    lightfieldGeometry->setVertexArray();
    
    
    lightfieldGeode->addDrawable( lightfieldGeometry );
    lightfieldGroup->addChild( lightfieldGeode );
    
    return lightfieldGroup;
}