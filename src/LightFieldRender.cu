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

texture < float, 2 > _lightfieldTexture;

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
int intersectQuad( Ray ray, Quad quad, float3* intersectedPoint )
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

__global__
void d_render( uint* d_output, float* d_depthBuffer, int canvasWidth, int canvasHeight, bool isDebugOn )
{
    
}


// LightFieldRender
float LightFieldRender::renderKernel( dim3 gridSize, dim3 blockSize, uint* d_output, float* d_depthBuffer )
{
    // Repassando parametros para o kernel...
    cudaEvent_t clock = CUDAManager::getInstance()->startClock();

    d_render <<< gridSize, blockSize >>> ( d_output, d_depthBuffer, _screenWidth, _screenHeight, true );

    return CUDAManager::getInstance()->stopClock( clock );
}

void LightFieldRender::initKernelParameters()
{    
    CUDAManager::getInstance()->collectError(
        cudaMemcpyToSymbol( _cudaKernelParameters, ( void* ) &_kernelParameters, sizeof( KernelParameters ) ) );
}