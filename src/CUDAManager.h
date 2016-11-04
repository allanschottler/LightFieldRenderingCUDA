#ifndef CUDAMANAGER_H_
#define CUDAMANAGER_H_

// Bibliotecas do sistema
#include <vector>
#include <string>

// Bibliotecas do 
#include "pthread.h"
//#include <mutex>
//#include "logger/vlog.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * Gerencia o CUDA.
 * Verifica se o sistema tem suporte a CUDA
 */
class CUDAManager
{
public:
    /**
     * Definicao de tipo que indica os diferentes estados
     * do gerenciador CUDA.
     */
    enum CUDA_MANAGER_STATE
    {
        /**
         * Indica que o gerenciador ainda nao foi devidamente
         * inicializado.
         */
        NOT_INITIALIZED = 0,

        /**
         * Indica que o sistema nao possui suporte a GPUs e
         * consequentemente nao podera utilizar os recursos
         * referentes ao emprego de placas.
         */
        NO_GPU_SUPPORT,

        /**
         * Indica que o sistema possui suporte a GPUs porem
         * o driver CUDA nao atende aos requisitos minimos
         * de versao (desatualizado//outdated).
         */
        DRIVER_OUTDATED,

        /**
         * Indica que o sistema possui suporte a GPUs, que o
         * driver atende aos requisitos minimos, mas a consulta
         * por dispositivos nao retornou nenhum.
         */
        GPUS_NOT_FOUND,

        /**
         * Indica que o sistema possui GPUs porem as mesmas
         * nao atendem ao perfil minimo necessario requerido.
         * (Ex: shared memory abaixo do requisitado, etc)
         */
        NO_VALID_GPUS,

        /**
         * Indica que o gerenciador foi inicializado com sucesso
         * e que atende a todos os requisitos minimos requiridos.
         */
        SUCCESSFULLY_INITIALIZED
    };

    /**
     * Versao minima necessaria para utilizacao da plataforma CUDA.
     */
    static const int MIN_CUDA_VERSION;

    /**
     * Minimo de memoria compartilhada minima necessaria.
     */
    static const unsigned long MIN_SHARED_MEMORY;

    /**
     * Armazena indice para dispositivo padrao de exibicao
     */
    static const int CUDA_DISPLAY_DEVICE;

    /**
     * Armazena a variavel de ambiente que indica que o sistema
     * nao possui GPUs disponiveis.
     */
    static const std::string NO_GPU_ENV_ID;

    /**
    * Retorna uma instancia unica do CudaManager
    * @return instancia unica do CudaManager
    */
    static CUDAManager* getInstance();

    /**
     * Desaloca a instancia unica do CudaManager
     */
    static void destroy();

    /**
     * Retorna verdadeiro se houver suporte a cuda
     */
    bool isGPUAvailable() const;

    /**
     * Retorna a quantidade de memoria livre na GPU.
     */
    unsigned long long getGPUFreeMemory();

    /**
     * Identifica qualquer erro retornado por uma funcao CUDA
     * @param[in] errorID - identificador do erro
     */
    void collectError( cudaError_t errorID );

    /**
     * Valida se a versao do CUDA e o maior ou igual ao minimo necessario.
     */
    bool validateCUDAVersion();

    /**
     * Retorna ultimo erro de CUDA
     * @param[out] errorMessage Mensagem de erro
     */
    bool getLastError( const char* errorMessage );

    /**
     * Comeca a contar o tempo de execucao e retorna o evento com o tempo inicial.
     * @return Evento com o tempo inicial
     */
    static cudaEvent_t startClock();

    /**
     * Para o relogio e imprime o tempo total gasto a partir do tempo inicial.
     * @param[in] start evento com o tempo inicial.
     * @param[in] parentFunction nome da funcao que disparou o relogio.
     */
    static void stopClock( cudaEvent_t start, const char* parentFunction );

    /**
     * Para o relogio e imprime o tempo total gasto a partir do tempo inicial.
     * @param[in] start evento com o tempo inicial.
     */
    static float stopClock( cudaEvent_t start );

    /**
     * Retorna se a placa atende alguns criterios necessarios
     * @param[in] deviceIndex Indice da GPU
     */
    bool isSupported( int deviceIndex );

    /**
     * Retorna se a placa//GPU selecionada tem o minimo necessario de
     * "shared memory" [memoria compartilhada].
     */
    bool hasSharedMemory();

    /**
     * Retorna se a placa//GPU selecionada suporta o array de textura 3D
     * independente de memória em placa, verificando apenas as dimensões
     * @param[in] i,j,k valores referentes as dimensoes do volume que se
     * deseja alocar em um array de textura 3D
     */
    bool verifyTexture3DArray( int i, int j, int k );

    /**
     * Faz requisicao de acesso a GPU e da lock caso a GPU esteja livre
     * @return verdadeiro se requisicao foi bem sucedida.
     */
    bool aquireGPU();

    /**
     * Libera lock de acesso a GPU
     */
    void releaseGPU();

    /**
     * @return Indice da GPU que esta em uso
     */
    int getCurrenDeviceIndex() const;

    /**
     * Usado para calculo de tamanho do grid usado pelo kernel.
     */
    static int iDivUp( int a, int b );

    /**
     * Seleciona dispositivo de indice 0
     */
    void setDisplayDevice( bool isToLog );

    /**
     * Retorna dispositivo padrao como ativo
     */
    void setDefaultDevice();

    /**
     * Indica se existe algum dispositivo com suporte a CUDA com as
     * especificacoes exigidas pelo .
     * @return Booleano que indica se existe um dispositivo CUDA.
     */
    bool hasCUDADevice() const;

    /**
     * Indica qual o estado corrente do gerenciador. Esses estados
     * podem apenas ser resgatados, pois os mesmos mudam de acordo
     * com as operacoes realizadas pelo gerenciador.
     */
    CUDA_MANAGER_STATE getState() const;

private:
    /**
     * Atributo para controle de Instancia Unica
     */
    static CUDAManager* _uniqueInstance;

    /**
     * Construtor
     */
    CUDAManager();

    /**
     * Destrutor
     */
    virtual ~CUDAManager();

    /**
     * Inicializa o gerenciador verificando se o sistema
     * possui suporte a CUDA e define o estado vigente do
     * mesmo.
     */
    void InitializeManager();

    /**
     * Retorna o numero de dispositivos com suporte a CUDA
     */
    int getDeviceCount();

    /**
     * Retorna numero de dispositivos com suporte a CUDA e
     * que atendem aos requisitos minimos necessarios.
     */
    int getSupportedDevices();

    /**
     * Pega indice de GPU com maior Gflops
     */
    int gpuGetMaxGflopsDeviceId();

    /**
     * Seleciona dispositivo por indice
     */
    void setDevice( int deviceIndex, bool isToLog = true );

    /**
     * Usado para o calculo de GPU com maior Gflops
     */
    static int convertSMVer2Cores( int major, int minor );

    /**
     * Armazena o estado corrente do gerenciador CUDA.
     */
    CUDA_MANAGER_STATE _state;

    /**
     * Armazena o numero de dispositivos CUDA detectados:
     * Total = suportados + nao_suportados.
     */
    int _deviceCount;

    /**
     * Armazena o numero de dispositivos CUDA detectados
     * e que atendem aos requisitos minimos necessarios.
     */
    int _supportedDevices;

    /**
     * Armazena o numero de dispositivos CUDA detectados
     * e que nao atendem aos requisitos minimos necessarios.
     */
    int _unsupportedDevices;

    /**
     * Armazena os indices dos dispositivos que atendem aos
     * requisitos minimos necessarios.
     */
    std::vector< int > _listOfSupportedDevices;

    /**
     * Verifica se a variavel de ambiente referente ao
     * suporte a GPU foi definida. Caso tenha sido, o
     * sistema nao possuira suporte a GPUs.
     */
    bool _hasGPUSupport;

    /**
     * Indica se o sistema tem suporte a CUDA, ou melhor, se
     * possui uma GPU que disponibilize suporte a plataforma CUDA.
     */
    bool _validGPUAvailable;

    /**
     * Semaforo para controle do acesso a GPU por exclusao mutua.
     */
    pthread_mutex_t _mutex;

    /**
     * Indice padrao da GPU que devara ser usada
     */
    int _defaultDeviceIndex;

    /**
     * Indice da GPU que esta sendo atualmente usada
     */
    int _currentDeviceIndex;

    /**
     * Variavel para controle do mecanismo de registro de eventos [LOG]
     */
    //vlog::CommandExecution _execution;
};
#endif /* CUDAMANAGER_H_ */
