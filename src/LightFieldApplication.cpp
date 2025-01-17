/* 
 * File:   LightFieldApplication.cpp
 * Author: allan
 */

#include "LightFieldApplication.h"
#include "LightFieldImageLoader.h"
#include "LightFieldDrawable.h"

#include <iostream>

namespace LightField
{
    
Application* Application::_instance = 0;

Application::Application() :
    _window( new MainWindow( "Light Field Rendering" ) ),
    _lightFieldImage( nullptr ),
    _lightFieldRender( nullptr ),
    _focalPlane( 1 ),
    _isDepthMap( false )
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


Application::~Application()
{
    if( _lightFieldRender )
        delete _lightFieldRender;
    
    delete _instance;    
    _instance = 0;
}


Application* Application::getInstance()
{
    if( !_instance )
        _instance = new Application();
    
    return _instance;
}


bool Application::loadLightField( std::string lightFieldHeader )
{    
    // Dispara threads
    if( !_lightFieldLoader.load( lightFieldHeader ) )
        return false;              
    
    return true;
}


void Application::createLightFieldNode()
{            
    // Limpa cena existente
    if( _lightFieldRender )
        delete _lightFieldRender; 
    
    _scene->removeChildren( 0, _scene->getNumChildren() );  
    
    // Pega imagem depois de threads terminarem
    _lightFieldImage = _lightFieldLoader.getLightFieldImage();            
    
    // Cria o render
    _lightFieldRender = new Render( _lightFieldImage );    
    _lightFieldRender->setFocalPlane( _focalPlane );
    
    // Cria o grafo
    osg::ref_ptr< osg::Drawable > lightFieldDrawable = new Drawable( _lightFieldRender );
        
    osg::ref_ptr< osg::Geode > lightfieldGeode = new osg::Geode;
    lightfieldGeode->addDrawable( lightFieldDrawable );
    
    osg::ref_ptr< osg::Group > lightfieldGroup = new osg::Group;
    lightfieldGroup->addChild( lightfieldGeode );    
    
    // Adiciona na cena
    _scene->addChild( lightfieldGroup );
    
    setFocalPlane( _focalPlane );
}


void Application::setFocalPlane( float focalPlane )
{
    _focalPlane = focalPlane;
        
    if( _lightFieldRender )
        _lightFieldRender->setFocalPlane( focalPlane );
}


void Application::setRenderAsDepthMap( bool isDepthMap )
{
    _isDepthMap = isDepthMap;
    
    if( _lightFieldRender )
        _lightFieldRender->setRenderAsDepthMap( isDepthMap );
}


void Application::printFPS()
{
    if( _lightFieldRender )
        std::cout << "FPS : " << _lightFieldRender->getFPS() << "\n";
}

}