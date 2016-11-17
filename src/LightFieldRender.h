/* 
 * File:   LightFieldRender.h
 * Author: allan
 */

#ifndef LIGHTFIELDRENDER_H
#define	LIGHTFIELDRENDER_H

#include <cuda.h>

#include "CUDAManager.h"
#include "LightFieldImage.h"

class LightFieldRender 
{
public:
    
    class KernelParameters
    {
    public:
        /* Ponto central do plano Near */
        float3 nearOrigin;

        /* Ponto central do plano Far */
        float3 farOrigin;

        /* Vetores utlizados para percorrer o plano near no algoritmo de tracado de raios */
        float3 uNear;
        float3 vNear;

        /* Vetores utlizados para percorer o plano far no algoritmo de tracado de raios */
        float3 uFar;
        float3 vFar;
        
        /* Valor que determina o plano Near em coordenada do olho. */
        float near;

        /* Valor que determina o plano Far em coordenada do olho. */
        float far;

        /* Parametros para a transformação inversa do valor do zbuffer */
        float zTransformParamA;
        float zTransformParamB;
        
        /* Dimensões da câmera plenóptica */
        int nCameraRows, nCameraCollumns;
        
        /* Distância do plano focal ao plano das câmeras */
        float focalPlane;
    };
    
    LightFieldRender( LightFieldImage* lightFieldImage );
    
    virtual ~LightFieldRender();
    
    void render();
    
    float renderKernel( dim3 gridSize, dim3 blockSize, uint* d_output, float* d_depthBuffer );
    
    void getBoundingBox( float& xMin, float& xMax, float& yMin, float& yMax, float& zMin, float& zMax );
    
    void setFocalPlane( float focalPlane );
    
    void computeFPS( float elapsedTime, float& fps );    
    
private:    
    
    void initLightFieldTexture( unsigned char* texels, int width, int height );
    
    void initPBO();
    
    void initCudaBuffers( uint*& d_output, float*& d_depthBuffer );
        
    void cleanCudaBuffers( float*& d_depthBuffer );
    
    void initKernelParameters();
    
    void updateParameters();
    
    void debugInfo();
    

    LightFieldImage* _lightFieldImage;
    
    /* Buffer intermediário para passagem de dados para a placa */
    unsigned char* _lightFieldTexels;
    
    /* Array com os valores do volume (recebe textura 3D) */
    cudaArray* _lightFieldArray;
    
    /* Estrutura de parâmetros passados para a GPU */
    KernelParameters _kernelParameters;

    /* Dimensões do canvas de desenho */
    int _screenWidth;
    int _screenHeight;

    /* Pixel Buffer Object */
    unsigned int _outPBO;

    /* Textura que recebe PBO pro render da tela */
    unsigned int _outTexture;

    /* Buffer de saída do kernel */
    //float* _outBuffer;
    
    /* Buffer de profundidade do OpenGL */
    float* _depthBuffer;    
        
    /* Calculo de fps */
    float _fps;
    unsigned int _frameCount;
    time_t _startTime;
    time_t _endTime;
};

#endif	/* LIGHTFIELDRENDER_H */

