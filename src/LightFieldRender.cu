/* 
 * File:   LightFieldRender.cu.cpp
 * Author: allan
 */

#include <cmath>
#include <iostream>

#include "LightFieldRender.h"
#include "helper_math.h"
#include "CUDAManager.h"

__constant__ LightFieldRender::KernelParameters _cudaKernelParameters;

texture< float4, 2 > _lightfieldTexture;

struct Ray
{
    float3 origin;
    float3 direction;
};

struct Quad
{
    float3 tlPoint; // top left s1
    float3 trPoint; // top right s2
    float3 blPoint; // bottom left s3
};

// AUX
__device__
bool intersectQuad( Ray ray, Quad quad, float3* intersectedPoint )
{
    float3 dS21 = quad.trPoint - quad.tlPoint;
    float3 dS31 = quad.blPoint - quad.tlPoint;
    float3 n = cross( dS21, dS31 );

    float ndotdR = dot( n, ray.direction );

    if( fabs( ndotdR ) < 1e-6f ) // tolerance
        return false;

    float t = dot( -n, (ray.origin - quad.tlPoint) ) / ndotdR;
    *intersectedPoint = ray.origin + ray.direction * t;

    float3 dMS1 = *intersectedPoint - quad.tlPoint; 
    float u = dot( dMS1, dS21 );
    float v = dot( dMS1, dS31 );

    return ( u >= 0.0f && u <= dot( dS21, dS21 ) && v >= 0.0f && v <= dot( dS31, dS31 ) );
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
void trace( Quad& quad, float3& hitPoint, float4& collectedColor )
{    
    // Lê a textura
    collectedColor = tex2D( _lightfieldTexture, 
                          map( hitPoint.x, quad.blPoint.x, quad.trPoint.x, 0, 1 ),
                          map( hitPoint.y, quad.blPoint.y, quad.trPoint.y, 0, 1 ) );
}

__global__
void d_render( uint* d_output, float* d_depthBuffer, int canvasWidth, int canvasHeight, bool isDebugOn )
{
    //Origem do raio em coordenadas de tela.
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if( ( x >= canvasWidth ) || ( y >= canvasHeight ) )
    {
        return;
    }

    //Mapeia x e y para o intervalo [-1, 1]
    float u = ( x / ( float ) canvasWidth ) * 2.0f - 1.0f;
    float v = ( y / ( float ) canvasHeight ) * 2.0f - 1.0f;

    float nearX, farX;
    float nearY, farY;
    float nearZ, farZ;
    setNearAndFar( u, v, nearX, farX, nearY, farY, nearZ, farZ );

    //Cria o quad
    float3 blPoint( make_float3( 0, 0, 0 ) );
    float3 tlPoint( make_float3( 0, _cudaKernelParameters.nCameraRows, 0 ) );
    float3 trPoint( make_float3( _cudaKernelParameters.nCameraCollumns, _cudaKernelParameters.nCameraRows, 0 ) );

    Quad quad;
    quad.blPoint = blPoint;
    quad.tlPoint = tlPoint;
    quad.trPoint = trPoint;
    
    Ray eyeRay;
    eyeRay.origin = make_float3( nearX, nearY, nearZ );
    eyeRay.direction = make_float3( farX - nearX, farY - nearY, farZ - nearZ );

    // Acha a interseção com a bounding box
    float3 hitPoint;
    bool hit = intersectQuad( eyeRay, quad, &hitPoint );

    // Para se o raio não interceptou a bounding box.
    if( !hit )
        return;

    // Valor de cor acumulada durante o traçado de raios.
    float4 collectedColor = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );

    // Traça o raio    
    trace( quad, hitPoint, collectedColor );

    // Grava saidas
    d_depthBuffer[ y * canvasWidth + x ] = 0; // TODO
    d_output[ y * canvasWidth + x ] = rgbaFloatToInt( collectedColor );
}


// LightFieldRender
float LightFieldRender::renderKernel( dim3 gridSize, dim3 blockSize, uint* d_output, float* d_depthBuffer )
{
    // Repassando parametros para o kernel...
    cudaEvent_t clock = CUDAManager::getInstance()->startClock();

    d_render <<< gridSize, blockSize >>> ( d_output, d_depthBuffer, _screenWidth, _screenHeight, true );

    return CUDAManager::getInstance()->stopClock( clock );
}


void LightFieldRender::initLightFieldTexture( float* texels, int width, int height )
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc< float4 >();

    CUDAManager::getInstance()->collectError( 
        cudaMallocArray( &_lightFieldArray, &channelDesc, width, height ) );    
    
    CUDAManager::getInstance()->collectError( 
        cudaMemcpy2DToArray( _lightFieldArray, 0, 0, texels, width * sizeof( float4 ), 
        width * sizeof( float4 ), height, cudaMemcpyHostToDevice ) );
        
    // Inicializa os parametros de textura
    _lightfieldTexture.normalized = true;                       
    _lightfieldTexture.filterMode = cudaFilterModeLinear;       
    _lightfieldTexture.addressMode[ 0 ] = cudaAddressModeClamp; 
    _lightfieldTexture.addressMode[ 1 ] = cudaAddressModeClamp; 

    // Associa o array a textura
    CUDAManager::getInstance()->collectError( 
        cudaBindTextureToArray( _lightfieldTexture, _lightFieldArray, channelDesc ) );
}


void LightFieldRender::initKernelParameters()
{    
    CUDAManager::getInstance()->collectError(
        cudaMemcpyToSymbol( _cudaKernelParameters, ( void* ) &_kernelParameters, sizeof( KernelParameters ) ) );
}