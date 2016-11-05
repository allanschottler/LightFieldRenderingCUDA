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
    };
    
    LightFieldRender( LightFieldImage* lightFieldImage );
    
    virtual ~LightFieldRender();
    
private:

    LightFieldImage* _lightFieldImage;
    
    /* Buffer intermediário para passagem de dados para a placa */
    float* _texels;

    /* Estrutura de parâmetros passados para a GPU */
    KernelParameters _kernelParameters;

    /* Dimensões do canvas de desenho */
    int _screenWidth;
    int _screenHeight;

    /* Pixel Buffer Object */
    unsigned int _pbo;

    /* Textura que recebe PBO pro render da tela */
    unsigned int _tex;

    /* Buffer de profundidade do OpenGL */
    float* _depthBuffer;

    /* Array com os valores do volume (recebe textura 3D) */
    cudaArray* _d_volumeArray;
};

#endif	/* LIGHTFIELDRENDER_H */

