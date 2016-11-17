/* 
 * File:   LightFieldApplication.cpp
 * Author: allan
 */

#include "LightFieldApplication.h"
#include "LightFieldImageLoader.h"
#include "LightFieldDrawable.h"

LightFieldApplication* LightFieldApplication::_instance = 0;

LightFieldApplication::LightFieldApplication() :
    _window( new MainWindow( "Light Field Rendering" ) ),
    _lightFieldImage( nullptr ),
    _lightFieldRender( nullptr ),
    _focalPlane( 1 )
{    
    // Configurações de cena
    _scene = new osg::Group;
    _scene->setDataVariance( osg::Object::DYNAMIC );
    _scene->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );             
    
    // Configurações de câmera
    float x = 17;
    float y = 17;
    
    osg::ref_ptr< osgGA::TrackballManipulator > manipulator = new osgGA::TrackballManipulator();
    _window->getCanvas().setCameraManipulator( manipulator );    
    _window->getCanvas().getCamera()->setClearColor( osg::Vec4( .0f, .0f, .0f, 1.f ) );
    _window->getCanvas().getCameraManipulator()->setHomePosition( 
        osg::Vec3d( x/2, y/2, 2*x ), 
        osg::Vec3d( x/2, y/2, 0. ), 
        osg::Vec3d( 0., 1., 0. ) );    
    
    // Parametros do spinbutton de plano focal
    _window->setFocalPlaneSpinAdjustment( 1, 9999., 1. );
    
    _window->getCanvas().setSceneData( _scene );
    _window->show(); 
}


LightFieldApplication::~LightFieldApplication()
{
    if( _lightFieldRender )
        delete _lightFieldRender;
    
    if( _lightFieldImage )
        delete _lightFieldImage;
    
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
    // Dispara threads
    if( !_lightFieldLoader.load( lightFieldHeader ) )
        return false;        
    
    return true;
}


void LightFieldApplication::createLightFieldNode()
{    
    // Pega imagem depois de threads terminarem
    _lightFieldImage = _lightFieldLoader.getLightFieldImage();            
    
    // Cria o render
    _lightFieldRender = new LightFieldRender( _lightFieldImage );    
    _lightFieldRender->setFocalPlane( _focalPlane );
    
    // Cria o grafo
    osg::ref_ptr< osg::Drawable > lightFieldDrawable = new LightFieldDrawable( _lightFieldRender );
        
    osg::ref_ptr< osg::Geode > lightfieldGeode = new osg::Geode;
    lightfieldGeode->addDrawable( lightFieldDrawable );
    
    osg::ref_ptr< osg::Group > lightfieldGroup = new osg::Group;
    lightfieldGroup->addChild( lightfieldGeode );    
    
    // Adiciona na cena
    _scene->addChild( lightfieldGroup );
    
    setFocalPlane( _focalPlane );
}


void LightFieldApplication::setFocalPlane( float focalPlane )
{
    _focalPlane = focalPlane;
        
    if( _lightFieldRender )
        _lightFieldRender->setFocalPlane( focalPlane );
}