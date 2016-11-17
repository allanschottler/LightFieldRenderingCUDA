/* 
 * File:   LightFieldRender.cu.cpp
 * Author: allan
 */

#include <cmath>
#include <iostream>
#include <stdio.h>

#include "LightFieldRender.h"
#include "helper_math.h"
#include "CUDAManager.h"

namespace LightField
{
__constant__ Render::KernelParameters _cudaKernelParameters;
}

texture< uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat > _lightfieldTexture;

using namespace LightField;

struct Ray
{
    float3 origin;
    float3 direction;
};

struct Quad
{
    float3 tlPoint; // top left
    float3 trPoint; // top right
    float3 blPoint; // bottom left
};

struct Plane
{
    float3 normal;
    float3 point;
};

// Auxiliares
__device__
bool intersectPlane( Ray ray, Plane plane, float3* intersectedPoint )
{    
    // Testa se acerta plano
    float ndotdR = dot( plane.normal, ray.direction );

    if( fabs( ndotdR ) < 1e-6f ) // tolerance
        return false;
    
    // Calcula ponto de interseção
    float t = dot( -plane.normal, ( ray.origin - plane.point ) ) / ndotdR;
    *intersectedPoint = ray.origin + ray.direction * t;
    
    return t >= 0;
}

__device__
bool intersectQuad( Ray ray, Quad quad, float3* intersectedPoint )
{
    // Define normal
    float3 dS21 = quad.trPoint - quad.tlPoint;
    float3 dS31 = quad.blPoint - quad.tlPoint;
    float3 n = cross( dS21, dS31 );
    
    Plane plane;
    plane.normal = n;
    plane.point = quad.tlPoint;

    if( intersectPlane( ray, plane, intersectedPoint ) )
    {
        // Verifica se interseção está contido no quad
        float3 dMS1 = *intersectedPoint - quad.tlPoint; 
        float u = dot( dMS1, dS21 );
        float v = dot( dMS1, dS31 );

        return ( u >= 0.0f && u <= dot( dS21, dS21 ) && v >= 0.0f && v <= dot( dS31, dS31 ) );
    }
    
    return false;
}

__device__
unsigned int rgbaFloatToInt( float4 rgba )
{
    rgba.x = __saturatef( rgba.x ); 
    rgba.y = __saturatef( rgba.y );
    rgba.z = __saturatef( rgba.z );
    rgba.w = __saturatef( rgba.w );

    typedef unsigned int uint;

    return ( uint( rgba.w * 255 ) << 24 ) | ( uint( rgba.z * 255 ) << 16 ) |
           ( uint( rgba.y * 255 ) << 8 ) | uint( rgba.x * 255 );  
}

__device__
inline float map( float x, float in_min, float in_max, float out_min, float out_max )
{
    return ( x - in_min ) * ( out_max - out_min ) / ( in_max - in_min ) + out_min;
}

__device__
void setNearAndFar( float u, float v, float& nearX, float& farX, float& nearY, float& farY, float& nearZ, float& farZ )
{
    // Calcula os pontos extremos do raio em coordenadas de mundo.
    nearX = _cudaKernelParameters.nearOrigin.x + u * _cudaKernelParameters.uNear.x + v * _cudaKernelParameters.vNear.x;
    nearY = _cudaKernelParameters.nearOrigin.y + u * _cudaKernelParameters.uNear.y + v * _cudaKernelParameters.vNear.y;
    nearZ = _cudaKernelParameters.nearOrigin.z + u * _cudaKernelParameters.uNear.z + v * _cudaKernelParameters.vNear.z;

    farX = _cudaKernelParameters.farOrigin.x + u * _cudaKernelParameters.uFar.x + v * _cudaKernelParameters.vFar.x;
    farY = _cudaKernelParameters.farOrigin.y + u * _cudaKernelParameters.uFar.y + v * _cudaKernelParameters.vFar.y;
    farZ = _cudaKernelParameters.farOrigin.z + u * _cudaKernelParameters.uFar.z + v * _cudaKernelParameters.vFar.z;
}

__device__
void getNeighbourCameras( float3 hitPoint, float3& c0, float3& c1, float3& c2, float3& c3 )
{    
    c0.x = floor( hitPoint.x );
    c0.y = floor( hitPoint.y );
        
    c1.x = c0.x + 1;
    c1.y = c0.y;
        
    c2.x = c0.x;
    c2.y = c0.y + 1;
        
    c3.x = c0.x + 1;
    c3.y = c0.y + 1;
}

__device__
void mapWorldToTexCoord( Plane& focalPlane, float3 fg, float3 st, float2& uv )
{    
    // Passo de uma microimagem em texcoord
    float sStep = 1. / _cudaKernelParameters.nCameraCollumns;
    float tStep = 1. / _cudaKernelParameters.nCameraRows;  
    
    // Semelhança de triangulos
    float3 stProj = make_float3( st.x, st.y, focalPlane.point.z );
    float3 stProjToFg = fg - stProj;
    float distanceToFocalPlane = length( stProj - st );
    float3 factor = stProjToFg / distanceToFocalPlane;
    
    // Obtem texcoord do raio para a camera em st
    uv.x = ( st.x + .5 + factor.x ) * sStep;
    uv.y = ( st.y + .5 + factor.y ) * tStep;    
}

__device__
void trace( Ray& ray, Quad& quad, Plane& focalPlane, float3& hitPoint, float4& collectedColor )
{    
    float3 hitPointPlane;
    bool hit = intersectPlane( ray, focalPlane, &hitPointPlane );
    
    if( !hit ) 
        return; // Nunca deveria cair aqui        
    
    float3 c0 = make_float3( 0., 0., 0. );
    float3 c1 = make_float3( 0., 0., 0. );
    float3 c2 = make_float3( 0., 0., 0. );
    float3 c3 = make_float3( 0., 0., 0. );
    
    // Obtem coordenada no mundo das camera vizinhas
    getNeighbourCameras( hitPoint, c0, c1, c2, c3 );
    
    // Mapeia cada raio para texcoord
    float2 uv0, uv1, uv2, uv3;
    mapWorldToTexCoord( focalPlane, hitPointPlane, c0, uv0 );
    mapWorldToTexCoord( focalPlane, hitPointPlane, c1, uv1 );
    mapWorldToTexCoord( focalPlane, hitPointPlane, c2, uv2 );
    mapWorldToTexCoord( focalPlane, hitPointPlane, c3, uv3 );
    
    // Obtem cores de cada camera
    float4 color0, color1, color2, color3;
    color0 = tex2D( _lightfieldTexture, uv0.x, uv0.y );
    color1 = tex2D( _lightfieldTexture, uv1.x, uv1.y );
    color2 = tex2D( _lightfieldTexture, uv2.x, uv2.y );
    color3 = tex2D( _lightfieldTexture, uv3.x, uv3.y );
    
    // Blend linear
    collectedColor = lerp( lerp( lerp( color0, color1, .5 ), color2, .5 ), color3, .5 );
}

__global__
void d_render( uint* d_output, float* d_depthBuffer, int canvasWidth, int canvasHeight, bool isDebugOn )
{
    // Origem do raio em coordenadas de tela.
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= canvasWidth || y >= canvasHeight )
        return;

    // Mapeia x e y para o intervalo [-1, 1]
    float u = ( x / ( float ) canvasWidth ) * 2.0f - 1.0f;
    float v = ( y / ( float ) canvasHeight ) * 2.0f - 1.0f;

    // Obtem raios de near a far
    float nearX, farX;
    float nearY, farY;
    float nearZ, farZ;
    setNearAndFar( u, v, nearX, farX, nearY, farY, nearZ, farZ );
    
    float3 farPoint = make_float3( farX, farY, farZ );
            
    // Cria o quad
    Quad quad;
    quad.tlPoint = make_float3( 0, 0, 0 );
    quad.blPoint = make_float3( 0, _cudaKernelParameters.nCameraRows - 1, 0 );
    quad.trPoint = make_float3( _cudaKernelParameters.nCameraCollumns - 1, 0, 0 );
    
    // Cria o raio
    Ray eyeRay;
    eyeRay.origin = make_float3( nearX, nearY, nearZ );
    eyeRay.direction = farPoint - eyeRay.origin;
    
    // Cria o plano focal
    Plane focalPlane;
    focalPlane.normal = normalize( cross( quad.blPoint, quad.trPoint ) );
    focalPlane.point  = make_float3( 0, 0, _cudaKernelParameters.focalPlane );
    
    // Acha a interseção com o quad
    float3 hitPoint;
    float4 collectedColor = make_float4( 0, 0, 0, 0 );
        
    bool hit = intersectQuad( eyeRay, quad, &hitPoint );
    
    // Para se o raio não interceptou o quad.
    if( !hit )
        return;
    
    // Traça o raio    
    trace( eyeRay, quad, focalPlane, hitPoint, collectedColor );

    // Calcula profundidade virtual
    float depth = length( hitPoint - eyeRay.origin ) / length( farPoint - eyeRay.origin );
    
    // Grava saidas
    d_depthBuffer[ y * canvasWidth + x ] = depth;    
    d_output[ y * canvasWidth + x ] = rgbaFloatToInt( collectedColor );
}


// LightFieldRender
namespace LightField
{

float Render::renderKernel( dim3 gridSize, dim3 blockSize, uint* d_output, float* d_depthBuffer )
{
    // Repassando parametros para o kernel...
    cudaEvent_t clock = CUDAManager::getInstance()->startClock();

    d_render <<< gridSize, blockSize >>> ( d_output, d_depthBuffer, _screenWidth, _screenHeight, true );

    return CUDAManager::getInstance()->stopClock( clock );
}

void Render::initLightFieldTexture( unsigned char* texels, int width, int height )
{
    // Aloca array da textura
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc< uchar4 >();

    CUDAManager::getInstance()->collectError( 
        cudaMallocArray( &_lightFieldArray, &channelDesc, width, height ) );    
    
    CUDAManager::getInstance()->collectError( 
        cudaMemcpy2DToArray( _lightFieldArray, 0, 0, texels, width * sizeof( uchar4 ), 
        width * sizeof( uchar4 ), height, cudaMemcpyHostToDevice ) );
        
    // Inicializa os parametros de textura
    _lightfieldTexture.normalized = true;                       
    _lightfieldTexture.filterMode = cudaFilterModeLinear;       
    _lightfieldTexture.addressMode[ 0 ] = cudaAddressModeClamp; 
    _lightfieldTexture.addressMode[ 1 ] = cudaAddressModeClamp; 

    // Associa o array a textura
    CUDAManager::getInstance()->collectError( 
        cudaBindTextureToArray( _lightfieldTexture, _lightFieldArray, channelDesc ) );
}

void Render::initKernelParameters()
{    
    // Inicializa parametros do kernel
    CUDAManager::getInstance()->collectError(
        cudaMemcpyToSymbol( _cudaKernelParameters, ( void* ) &_kernelParameters, sizeof( KernelParameters ) ) );
}

}