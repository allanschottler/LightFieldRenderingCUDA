/* 
 * File:   LightFieldRender.cpp
 * Author: allan
 */

#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <ctime>
#include <math.h>
#include <cfloat>
#include <iostream>

#include "LightFieldRender.h"
#include "CUDAManager.h"

cudaGraphicsResource* _cudaPBOResource;

LightFieldRender::LightFieldRender( LightFieldImage* lightFieldImage ) :
    _lightFieldImage( lightFieldImage )
{
}


LightFieldRender::~LightFieldRender() 
{
    CUDAManager::getInstance()->setDisplayDevice( false );

    if( _outPBO )
    {
        CUDAManager::getInstance()->collectError(
            cudaGraphicsUnregisterResource( _cudaPBOResource ) );
        glDeleteBuffersARB( 1, &_outPBO );
        glDeleteTextures( 1, &_outTexture );
    }

    CUDAManager::getInstance()->setDefaultDevice();

    if( _lightFieldTexels )
    {
        delete[] _lightFieldTexels;
    }

    if( _depthBuffer )
    {
        delete[] _depthBuffer;
    }
}


void LightFieldRender::render()
{
    if( !CUDAManager::getInstance()->aquireGPU() )
        return;
    
    CUDAManager::getInstance()->setDisplayDevice( false );

    _frameCount++;
    _startTime = time( NULL );

    // Recupera as dimensões da tela
    GLint viewport[ 4 ];
    glGetIntegerv( GL_VIEWPORT, viewport );

    _screenWidth = viewport[ 2 ];
    _screenHeight = viewport[ 3 ];

    initPBO();
    
    updateParameters();

    // Recupera a matriz modelView do OpenGL
    GLfloat modelView[ 16 ];
    glGetFloatv( GL_MODELVIEW_MATRIX, modelView );

    dim3 blockSize( 16, 16 );
    dim3 gridSize = dim3( CUDAManager::getInstance()->iDivUp( _screenWidth, blockSize.x ),
                          CUDAManager::getInstance()->iDivUp( _screenHeight, blockSize.y ) );

    uint* d_output;
    float* d_depthBuffer;

    initCudaBuffers( d_output, d_depthBuffer );
    initKernelParameters();

    float elapsedTime = renderKernel( gridSize, blockSize, d_output, d_depthBuffer );

    CUDAManager::getInstance()->getLastError( "VolumeRender::render(): render_kernel failed" );

    cleanCudaBuffers( d_depthBuffer );

    // Desenha imagem do PBO
    glDisable( GL_DEPTH_TEST );

    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );

    glMatrixMode( GL_PROJECTION );
    glPushMatrix();
    glLoadIdentity();
    glOrtho( 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 );

    // Desenha utlizando textura
    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glLoadIdentity();

    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    // Copia dados do PBO para a textura.
    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, _outPBO );
    glBindTexture( GL_TEXTURE_2D, _outTexture );
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, _screenWidth, _screenHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );

    // Desenha um quad de textura.
    glEnable( GL_TEXTURE_2D );
    glBegin( GL_QUADS );
    glTexCoord2f( 0, 0 );
    glVertex2f( 0, 0 );
    glTexCoord2f( 1, 0 );
    glVertex2f( 1, 0 );
    glTexCoord2f( 1, 1 );
    glVertex2f( 1, 1 );
    glTexCoord2f( 0, 1 );
    glVertex2f( 0, 1 );
    glEnd();

    glDisable( GL_TEXTURE_2D );
    glBindTexture( GL_TEXTURE_2D, 0 );
    glEnable( GL_DEPTH_TEST );

    glMatrixMode( GL_PROJECTION );
    glPopMatrix();

    glMatrixMode( GL_MODELVIEW );
    glPopMatrix();

    glDisable( GL_BLEND );

    computeFPS( elapsedTime, _fps );

    CUDAManager::getInstance()->releaseGPU();
    CUDAManager::getInstance()->setDefaultDevice();
}


void LightFieldRender::getBoundingBox( float& xMin, float& xMax, float& yMin, float& yMax, float& zMin, float& zMax )
{
    xMin = xMax = yMin = yMax = zMin = zMax = 0;
}


void LightFieldRender::initPBO()
{
    if( _outPBO )
    {
        // unregister this buffer object from CUDA C
        CUDAManager::getInstance()->collectError(
            cudaGraphicsUnregisterResource( _cudaPBOResource ) );

        // Deleta o buffer antigo
        glDeleteBuffersARB( 1, &_outPBO );
        glDeleteTextures( 1, &_outTexture );
    }

    // Cria pixel buffer object  para renderizar
    glGenBuffersARB( 1, &_outPBO );
    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, _outPBO );
    glBufferDataARB( GL_PIXEL_UNPACK_BUFFER_ARB, _screenWidth * _screenHeight * sizeof( GLubyte ) * 4,
                     0, GL_STREAM_DRAW_ARB );
    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
	
    // Registra o buffer object
    CUDAManager::getInstance()->collectError(
        cudaGraphicsGLRegisterBuffer( &_cudaPBOResource, _outPBO, cudaGraphicsMapFlagsWriteDiscard ) );

    // Define parametros da textura do OpenGL
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glPixelStorei( GL_UNPACK_ROW_LENGTH, _screenWidth );
    glPixelStorei( GL_UNPACK_IMAGE_HEIGHT, _screenHeight );
    glPixelStorei( GL_UNPACK_SKIP_PIXELS, 0 );
    glPixelStorei( GL_UNPACK_SKIP_ROWS, 0 );
    glPixelStorei( GL_UNPACK_SKIP_IMAGES, 0 );

    // Criar textura para renderizar
    glGenTextures( 1, &_outTexture );
    glBindTexture( GL_TEXTURE_2D, _outTexture );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, _screenWidth, _screenHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glBindTexture( GL_TEXTURE_2D, 0 );
}

    
void LightFieldRender::initCudaBuffers( uint*& d_output, float*& d_depthBuffer )
{    
    // Mapeia o PBO para o CUDA
    CUDAManager::getInstance()->collectError(
        cudaGraphicsMapResources( 1, &_cudaPBOResource, 0 ) );

    size_t num_bytes;
    CUDAManager::getInstance()->collectError(
        cudaGraphicsResourceGetMappedPointer( ( void** )&d_output, &num_bytes,
                                              _cudaPBOResource ) );

    CUDAManager::getInstance()->collectError(
        cudaMemset( d_output, 0, _screenWidth * _screenHeight * 4 ) );

    // Aloca o buffer de profundidade na placa
    CUDAManager::getInstance()->collectError(
        cudaMalloc( ( void** ) &d_depthBuffer, _screenWidth * _screenHeight * sizeof( float ) ) );
       
    if( _depthBuffer )
    {
        delete[] _depthBuffer;
    }
    
    _depthBuffer = new float[ _screenWidth * _screenHeight ];
    
    glReadPixels( 0, 0, _screenWidth, _screenWidth, GL_DEPTH_COMPONENT, GL_FLOAT, _depthBuffer );

    CUDAManager::getInstance()->collectError(
        cudaMemcpy( d_depthBuffer, _depthBuffer, _screenWidth * _screenHeight * 4, cudaMemcpyHostToDevice ) );    
}


void LightFieldRender::cleanCudaBuffers( float*& d_depthBuffer )
{     
    CUDAManager::getInstance()->collectError(
        cudaMemcpy( _depthBuffer, d_depthBuffer, _screenWidth * _screenHeight * 4, cudaMemcpyDeviceToHost ) );

    // Depth retornado pelo kernel
//    if( _isDebugOn )
//    {
//        debugInfo( _depthBuffer );
//    }

    CUDAManager::getInstance()->collectError(
        cudaGraphicsUnmapResources( 1, &_cudaPBOResource, 0 ) );

    CUDAManager::getInstance()->collectError(
        cudaFree( d_depthBuffer ) );
}


void LightFieldRender::updateParameters()
{
    // Matriz modelview do 
    GLdouble modelView[ 16 ];
    // Matiz de projeção do 
    GLdouble projection[ 16 ];
    // View port da camera
    GLint viewPort[ 4 ];

    // Recupera a matriz modelview do opengl
    glGetDoublev( GL_MODELVIEW_MATRIX, modelView );
    // Recupera a matriz de projeção do opengl
    glGetDoublev( GL_PROJECTION_MATRIX, projection );
    // Recupera a view port
    glGetIntegerv( GL_VIEWPORT, viewPort );

    int width = viewPort[ 2 ];
    int height = viewPort[ 3 ];

    // Recupera o ponto no centro da tela no near e no far
    double nearOriginXDouble, nearOriginYDouble, nearOriginZDouble;
    double farOriginXDouble, farOriginYDouble, farOriginZDouble;

    gluUnProject( width / 2.0f, height / 2.0f, 0.0, modelView, projection, viewPort,
                  &nearOriginXDouble, &nearOriginYDouble, &nearOriginZDouble );
    gluUnProject( width / 2.0f, height / 2.0f, 1.0, modelView, projection, viewPort,
                  &farOriginXDouble, &farOriginYDouble, &farOriginZDouble );

    _kernelParameters.nearOrigin.x = ( float ) nearOriginXDouble;
    _kernelParameters.nearOrigin.y = ( float ) nearOriginYDouble;
    _kernelParameters.nearOrigin.z = ( float ) nearOriginZDouble;

    _kernelParameters.farOrigin.x = ( float ) farOriginXDouble;
    _kernelParameters.farOrigin.y = ( float ) farOriginYDouble;
    _kernelParameters.farOrigin.z = ( float ) farOriginZDouble;

    //Recupera pontos auxiliares para calcular o vetor uNear e uFar
    double auxNearX, auxNearY, auxNearZ;
    double auxFarX, auxFarY, auxFarZ;

    gluUnProject( width, height / 2.0, 0.0, modelView, projection, viewPort, &auxNearX, &auxNearY, &auxNearZ );
    gluUnProject( width, height / 2.0, 1.0, modelView, projection, viewPort, &auxFarX, &auxFarY, &auxFarZ );

    _kernelParameters.uNear.x = ( float )( auxNearX - nearOriginXDouble );
    _kernelParameters.uNear.y = ( float )( auxNearY - nearOriginYDouble );
    _kernelParameters.uNear.z = ( float )( auxNearZ - nearOriginZDouble );

    _kernelParameters.uFar.x = ( float )( auxFarX - _kernelParameters.farOrigin.x );
    _kernelParameters.uFar.y = ( float )( auxFarY - _kernelParameters.farOrigin.y );
    _kernelParameters.uFar.z = ( float )( auxFarZ - _kernelParameters.farOrigin.z );

    // Recupera pontos auxiliares para calcular o vetor vNear e vFar
    gluUnProject( width / 2.0f, height, 0.0, modelView, projection, viewPort, &auxNearX, &auxNearY, &auxNearZ );
    gluUnProject( width / 2.0f, height, 1.0, modelView, projection, viewPort, &auxFarX, &auxFarY, &auxFarZ );

    _kernelParameters.vNear.x = ( float )( auxNearX - nearOriginXDouble );
    _kernelParameters.vNear.y = ( float )( auxNearY - nearOriginYDouble );
    _kernelParameters.vNear.z = ( float )( auxNearZ - nearOriginZDouble );

    _kernelParameters.vFar.x = ( float )( auxFarX - farOriginXDouble );
    _kernelParameters.vFar.y = ( float )( auxFarY - farOriginYDouble );
    _kernelParameters.vFar.z = ( float )( auxFarZ - farOriginZDouble );

    /*
       Obtem a partir da matriz de projecao os parametros necessarios para a transformacao
       do valor do zbuffer para z olho.
       A = (far + near)/(far - near)
       B = 2 * far * near / (far - near)
    */
    _kernelParameters.zTransformParamA = -projection[ 14 ];
    _kernelParameters.zTransformParamB = projection[ 10 ];

    /*
       Calcula o near e o far dado que:
       zeye = d_parameters.zTransformParamA /
              ( zbuffer + d_parameters.zTransformParamB )
       para z entre -1.0 e 1.0 entao:
       zbuffer = -1 -> near
       zbuffer = 1 -> far
    */
    _kernelParameters.near = -projection[ 14 ] / ( projection[ 10 ] - 1.0 );
    _kernelParameters.far = _kernelParameters.zTransformParamA / ( _kernelParameters.zTransformParamB + 1.0 );
}


void LightFieldRender::computeFPS( float elapsedTime, float& fps )
{
    _endTime = time( NULL );
    fps = ( double ) 1 / ( elapsedTime / 1000.0f );
    _frameCount = 0;
}