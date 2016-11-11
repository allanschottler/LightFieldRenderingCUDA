#include <iostream>
#include <algorithm>
#include <string>
#include <sstream>
#include <cuda_gl_interop.h>

#include <mutex>

// Bibliotecas do 
//#include "GtkUtils.h"
#include "CUDAManager.h"

// Versao minima necessaria para utilizacao da plataforma CUDA.
const int CUDAManager::MIN_CUDA_VERSION = 6000;

// Inicializando controlador de instancia unica
CUDAManager* CUDAManager::_uniqueInstance = nullptr;

// Inicializando quantidade de memoria compartilhada minima
// necessaria a execucao dos metodos que empregam CUDA (48k).
const unsigned long CUDAManager::MIN_SHARED_MEMORY = 49152;

// Inicializando controlador de dispositivo de exibicao//display
const int CUDAManager::CUDA_DISPLAY_DEVICE = 0;

// ID da variavel de ambiente que indica ausencia de GPUs
const std::string CUDAManager::NO_GPU_ENV_ID = "_NO_GPU";


CUDAManager* CUDAManager::getInstance()
{
    if( !_uniqueInstance )
    {
        _uniqueInstance = new CUDAManager();
    }

    return _uniqueInstance;
}


void CUDAManager::destroy()
{
    if( _uniqueInstance )
    {
        delete _uniqueInstance;
        _uniqueInstance = nullptr;
    }
}


CUDAManager::CUDAManager() :
    _state( NOT_INITIALIZED ),
    _deviceCount( 0 ),
    _supportedDevices( 0 ),
    _unsupportedDevices( 0 ),
    _hasGPUSupport( true ),
    _validGPUAvailable( true ),
    _defaultDeviceIndex( -1 ),
    _currentDeviceIndex( -1 )
{
    // Inicializando o ponteiro do semaforo...
    pthread_mutex_init( &_mutex, NULL );

    // Verifica se a variavel de ambiente (que indica que
    // que o sistema nao possui GPUs) foi definida...
    const char* no_gpu_was_defined = getenv( NO_GPU_ENV_ID.c_str() );
    if( no_gpu_was_defined )
    {
        // Definindo estado do gerenciador...
        _hasGPUSupport = false;
        _validGPUAvailable = false;
        _state = NO_GPU_SUPPORT;

        // Registrando mensagem...
        std::string msg = "Status: NO_GPU_SUPPORT.";
		std::cout << msg << std::endl;
    }
    else
    {
        // Executa inicializacao do gerenciador.
        InitializeManager();
    }
}


CUDAManager::~CUDAManager()
{
    pthread_mutex_destroy( &_mutex );
}


void CUDAManager::InitializeManager()
{
    // Verifica se existe suporte a GPU atraves da
    // variavel de ambiente _NO_GPU...
    if( !_hasGPUSupport )
    {
        // Definindo estado do gerenciador...
        _state = NO_GPU_SUPPORT;
        _validGPUAvailable = false;

        // Registrando mensagem...
		std::string msg = "Status: NO_GPU_SUPPORT.";
		std::cout << msg << std::endl;

        // Retorno imediato...
        return;
    }

    // Verifica se o driver instalado atende aos
    // requisitos minimos pedidos pelo gerenciador
    if( !validateCUDAVersion() )
    {
        // Definindo estado do gerenciador...
        _state = DRIVER_OUTDATED;
        _validGPUAvailable = false;

        // Registrando mensagem...
        std::string msg = "Status: DRIVER_OUTDATED.";
		std::cout << msg << std::endl;

        // Retorno imediato...
        return;
    }

    // Verifica se algum dispositivo CUDA foi encontrado...
    _deviceCount = getDeviceCount();
    if( _deviceCount == 0 )
    {
        // Definindo estado do gerenciador...
        _state = GPUS_NOT_FOUND;
        _validGPUAvailable = false;

        // Registrando mensagem...
        std::string msg = "Status: GPUS_NOT_FOUND.";
		std::cout << msg << std::endl;

        // Retorno imediato...
        return;
    }

    // Verifica a quantidade de dispositivos com
    // suporte a CUDA e aos requisitos minimos...
    if( _deviceCount > 0 )
    {
        // Resgata dispositivos que atendem aos requisitos minimos...
        _supportedDevices = getSupportedDevices();

        // Contabiliza dispositivos nao-suportados...
        _unsupportedDevices = _deviceCount - _supportedDevices;

        // Registra quantidade de dispositivos...
		std::stringstream ss;
		ss << "Quantidade de dispositivos:" <<
			_supportedDevices << "(validos), " <<
			_unsupportedDevices << "(invalidos).";

		std::cout << ss.str() << std::endl;

        // Verifica quantidade de dispositivos suportados...
        if( _supportedDevices == 0 )
        {
            // Definindo estado do gerenciador...
            _state = NO_VALID_GPUS;
            _validGPUAvailable = false;

            // Registrando mensagem...
            std::string msg = "Status: NO_VALID_GPUS.";
			std::cout << msg << std::endl;

            // Retorno imediato...
            return;
        }

        // Indicando que o sistema possui pelo menos um
        // dispositivo valido que atende aos requisitos...
        _validGPUAvailable = true;

        // Resgata indice do dispositivo valido com maior
        // capacidade em Gflops...
        _defaultDeviceIndex = gpuGetMaxGflopsDeviceId();

        // Executa uma segunda verificacao com base na
        // avaliacao do dispositivo com melhor performance:
        // caso nenhum dispositivo tenha sido encontrado,
        // invalida o estado do gerenciador...
        if( _defaultDeviceIndex == -1 )
        {
            // Definindo estado do gerenciador...
            _state = NO_VALID_GPUS;
            _validGPUAvailable = false;

            // Registrando mensagem...
            std::string msg = "Status: NO_VALID_GPUS. | Invalid Default Device Index.";
			std::cout << msg << std::endl;

            // Retorno imediato...
            return;
        }

        // Indica successo na inicializacao do dispositivo...
        _state = SUCCESSFULLY_INITIALIZED;

        // Define dispositivo utilizado...
        setDevice( _defaultDeviceIndex );

        // Registrando mensagem...
        std::string msg = "Status: SUCCESSFULLY_INITIALIZED!";
		std::cout << msg << std::endl;
    }
}


bool CUDAManager::isGPUAvailable() const
{
    // Verifica se existe suporte a CUDA e se ha
    // algum dispositivo suportado se encontra
    // atualmente disponivel...
    return _hasGPUSupport && _validGPUAvailable;
}


void CUDAManager::setDevice( int deviceIndex, bool isToLog )
{
    // Verifica se o gerenciador se encontra em um
    // estado valido ou de completa inicializacao ou se
    // o indice do dispositivo se encontra valido...
    if( _state != SUCCESSFULLY_INITIALIZED || deviceIndex < 0 )
    {
        // Emitindo mensagem na saida de erro...
        std::string msg = "%s :: Invalid device index -> %d\n";
        fprintf( stderr, msg.c_str(), __FUNCTION__, deviceIndex );

        // Registrando mensagem no log...
        //_execution.logMessageF( msg.c_str(), __FUNCTION__, deviceIndex );

        // Retorno imediato...
        return;
    }

    // Coleta erros de definicao de dispositivo...
    //std::cout << "Set device\n";
    collectError( cudaSetDevice( deviceIndex ) );
    _currentDeviceIndex = deviceIndex;

    // Verifica se e necessario registrar mensagem...
    if( isToLog )
    {
        // Resgata informacoes do dispositivo utilizado...
        cudaDeviceProp deviceProp;
        cudaError_t cudaResultCode = cudaGetDeviceProperties( &deviceProp, deviceIndex );

        // Verifica se a ultima operacao foi executada com sucesso...
        if( cudaResultCode == cudaSuccess )
        {
            // Registra dispositivo que sera utilizado...
			std::stringstream ss;
			ss << "Dispositivo CUDA escolhido: " << deviceProp.name << " -> GPU_Index: " << deviceIndex << ".";

			std::cout << ss.str() << std::endl;
        }
        else
        {
            // Registra mensagem de possivel erro...
			std::stringstream ss;
			ss << "Erro ao recuperar informacoes do dispositivo -> Index: " << deviceIndex << ".";

			std::cout << ss.str() << std::endl;
        }
    }
}


void CUDAManager::setDisplayDevice( bool isToLog )
{
    setDevice( CUDA_DISPLAY_DEVICE, isToLog );
}


void CUDAManager::setDefaultDevice()
{
    setDevice( _defaultDeviceIndex, false );
}


bool CUDAManager::hasCUDADevice() const
{
    return ( _hasGPUSupport ) && ( _supportedDevices > 0 );
}


int CUDAManager::getDeviceCount()
{
    // Inicializa variaveis auxiliares...
    int deviceCount = 0;

    // Resgata quantidade de dispositivos e verifica sucesso da
    // operacao de resgate. Registra mensagem caso necessario.
    cudaError_t cudaResultCode = cudaGetDeviceCount( &deviceCount );
    if( cudaResultCode != cudaSuccess )
    {
        // Mensagem de erro ao resgatar numero de dispositivos...
        std::string msg = "Erro ao recuperar o numero de dispositivos com suporte a CUDA.";
		std::cout << msg << std::endl;

        // Definindo numero de dispositivos...
        deviceCount = 0;
    }

    // Retorna quantidade de dispositivos...
    return deviceCount;
}


int CUDAManager::getSupportedDevices()
{
    // Inicializacao de variaveis auxiliares...
    int result = 0;

    // Verifica quantidade de dispositivos detectados...
    if( _deviceCount > 0 )
    {
        // Listagem dos dispositivos ranqueados...
        std::string msg = "Dispositivos com suporte a plataforma CUDA:";
		std::cout << msg << std::endl;

        for( int index = 0; index < _deviceCount; ++index )
        {
            // Registra dispositivos que atendem aos requisitos minimos
            if( isSupported( index ) )
            {
                ++result;
                _listOfSupportedDevices.push_back( index );
            }
        }
    }

    // Retorna quantidade de dispositivos suportados...
    return result;
}


bool CUDAManager::isSupported( int deviceIndex )
{
    // Resgata propriedades do dispositivo fornecido...
    cudaDeviceProp deviceProp;
    cudaError_t cudaResultCode = cudaGetDeviceProperties( &deviceProp, deviceIndex );

    // Verifica se houve algum erro...
    if( cudaResultCode != cudaSuccess )
    {
        // Define mensagem de erro...
        std::string msg = "Erro ao verificar o dispositivo de indice %d. ";
		std::stringstream ss;
		ss << "Erro ao verificar o dispositivo de indice " << deviceIndex << ".Dispositivo nao atende aos criterios necessarios minimos.";
		std::cout << ss.str() << std::endl;

        // Registra mensagem de erro...
        //_execution.logMessageF( msg.c_str(), deviceIndex );

        return false;
    }

    // Verifica se o dispositivo se encontra em modo proibido,
    // ou seja, nenhuma thread pode utilizar cudaSetDevice()
    // para este dispositivo...
    if( deviceProp.computeMode == cudaComputeModeProhibited )
    {
        // Registra mensagem no log...
        std::string msg = "O dispositivo (%s|Index:%d) se encontra em modo proibido!";
        //_execution.logMessageF( msg.c_str(), deviceProp.name, deviceIndex );
		std::cout << msg << std::endl;
        // Retorno imediato caso device esteja bloqueado...
        return false;
    }

    // Registra propriedades do dispositivo analisado...
    std::string msg = "Device: %s | GPU_Index: %d | Memory: %zu MB | SharedMemory Per Block: %d KB | Device Version: %d.%d | %s.";
    std::string device_type = ( deviceProp.integrated )?( "On-Board_Device" ):( "Off-Board_Device" );
	std::cout << msg << std::endl;
    /*_execution.logMessageF( msg.c_str(), deviceProp.name,
                            deviceIndex, deviceProp.totalGlobalMem / ( 1024 * 1024 ),
                            deviceProp.sharedMemPerBlock / 1024,
                            deviceProp.major, deviceProp.minor,
                            device_type.c_str() );*/

    // Exclui placas com arquitetura abaixo da 2.0 ou que tiverem menos
    // que 48k de shared memory...
    if( deviceProp.major < 2 || deviceProp.sharedMemPerBlock < MIN_SHARED_MEMORY )
    {
        // Define mensagem de aviso caso disponivel seja incompativel...
        std::string msg =
            "Dispositivo (%s|Index:%d) incompativel: Arch Version < 2.0 ou Shared Memory por bloco < 48k!";
		std::cout << msg << std::endl;
        //_execution.logMessageF( msg.c_str(), deviceProp.name, deviceIndex );

        // Retorno imediato, dispositivo incompativel...
        return false;
    }

    // Retorno do metodo: caso de sucesso,
    // dispositivo atende aos requisitos...
    return true;
}


bool CUDAManager::hasSharedMemory()
{
    // Resgata dispositivo atualmente utilizado...
    int deviceIndex;
    cudaError_t cudaResultCode = cudaGetDevice( &deviceIndex );

    // Define valor inicial de retorno do metodo...
    bool result = false;

    // Verifica se houve algum erro...
    if( cudaResultCode != cudaSuccess )
    {
        // Define mensagem de erro...
        std::string msg = "Placa selecionada nao possui memoria compartilhada suficiente.";
		std::cout << msg << std::endl;

        // Registra mensagem de erro...
        //_execution.logMessage( msg );

        // Definindo retorno...
        return result;
    }

    // Resgata propriedades do dispositivo fornecido...
    cudaDeviceProp deviceProp;
    cudaResultCode = cudaGetDeviceProperties( &deviceProp, deviceIndex );

    // Verifica se o dispositivo atende aos requisitos necessarios
    if( cudaResultCode == cudaSuccess &&
        deviceProp.sharedMemPerBlock >= MIN_SHARED_MEMORY )
    {
        result = true;
    }

    // Definindo retorno...
    return result;
}


bool CUDAManager::verifyTexture3DArray( int i, int j, int k )
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties( &deviceProp, _currentDeviceIndex );

    auto maxTex = deviceProp.maxTexture3D;
    /*_execution.logMessageF( "Maximo array de textura 3D no dispositivo %d = ( %d,%d,%d )\n",
                            _currentDeviceIndex, maxTex[ 0 ], maxTex[ 1 ], maxTex[ 2 ] );*/

    // Realiza a verificação dos valores i,j,k com relação aos valores do array de textura suportado pela GPU
    // Se ao menos um for maior retorna falso e isto implica que o array não poderá ser alocado
    if( ( i > maxTex[ 0 ] ) || ( j > maxTex[ 1 ] ) || ( k > maxTex[ 2 ] ) )
    {
        return false;
    }
    else
    {
        return true;
    }
}


unsigned long long CUDAManager::getGPUFreeMemory()
{
    std::size_t free, total;
    cudaError_t cudaResultCode = cudaMemGetInfo( &free, &total );

    if( cudaResultCode == cudaSuccess )
    {
        return free;
    }

    std::string msg = "Erro ao consultar quantidade de memoria da placa selecionada.";
	std::cout << msg << std::endl;

    return 0;
}


void CUDAManager::collectError( cudaError_t errorID )
{
    if( errorID == cudaSuccess )
    {
        return;
    }

    fprintf( stderr, "%s :: %s\n", __FUNCTION__, cudaGetErrorString( errorID ) );

    std::string msg = cudaGetErrorString( errorID );
	std::cout << msg << std::endl;
}


bool CUDAManager::validateCUDAVersion()
{
    // Inicializando variaveis...
    int cudaVersion = 0;
    bool result = true;

    // Resgata versao do driver residente na maquina...
    cudaError_t cudaResultCode = cudaDriverGetVersion( &cudaVersion );

    // Verificando resgate de versao...
    if( cudaResultCode != cudaSuccess )
    {
        // Registrando mensagem...
        std::string msg = "Erro ao obter versao do driver!";
		std::cout << msg << std::endl;
        //_execution.logMessage( msg );

        // Emitindo mensagem na saida de erro..
        //std::cout << "Validate cuda vers\n";
        collectError( cudaResultCode );

        // Retorno imediato...
        return false;
    }

    // Registra versao do driver no log...
    //_execution.logMessageF( "Versão da plataforma CUDA instalada: %d", cudaVersion );

    // Verifica se versao do driver atende o minimo...
    if( cudaVersion < MIN_CUDA_VERSION )
    {
        // Indica que o driver esta obsoleto...
        result = false;

        // Registra mensagem no LOG...
        std::string msg = "Versao do driver menor do que a minima necessaria.";
		std::cout << msg << std::endl;
    }

    // Retorno da verificacao...
    return result;
}


bool CUDAManager::getLastError( const char* errorMessage )
{
    cudaError_t err = cudaGetLastError();
    bool result = true;

    if( cudaSuccess != err )
    {
        result = false;
        fprintf( stderr, "%s(%i) : getLastError() CUDA error : %s : (%d) %s.\n",
                 __FILE__, __LINE__, errorMessage, ( int )err, cudaGetErrorString( err ) );

        char strError[ 500 ];
        sprintf( strError, " getLastError() CUDA error : %s : (%d) %s. \n",
                 errorMessage, ( int )err, cudaGetErrorString( err ) );

		std::cout << std::string(strError) << std::endl;
    }

    return result;
}


int CUDAManager::gpuGetMaxGflopsDeviceId()
{
    // Verifica dispositivos suportados...
    int device_count = _supportedDevices;
    if( device_count == 0 )
    {
        // Exibindo mensagem na saida padrao de erro...
        std::string msg = "CUDA error: no devices supporting CUDA :: %s.\n";
        fprintf( stderr, msg.c_str(), __FUNCTION__ );

        // Registrando mensagem no LOG...
        //_execution.logMessageF( msg.c_str(), __FUNCTION__ );

        // Retorno imediato [indice para dispositivo invalido]...
        return -1;
    }

    // Find the best major SM Architecture GPU device...
    int best_SM_arch = 0;

    // Looping para verificar a melhor 'best_SM_arch'...
    unsigned int myListSize = _listOfSupportedDevices.size();
    for( unsigned int index = 0; index < myListSize; ++index )
    {
        // Resgata indice do dispositivo suportado...
        int current_device = _listOfSupportedDevices[ index ];

        // Resgata informacoes do dispositivo...
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties( &deviceProp, current_device );

        // If this GPU is not running on Compute Mode prohibited,
        // then we can add it to the list...
        if( deviceProp.computeMode != cudaComputeModeProhibited )
        {
            if( deviceProp.major > 0 && deviceProp.major < 9999 )
            {
                best_SM_arch = (std::max)( best_SM_arch, deviceProp.major );
            }
        }
    }

    // Escolhe por padrao o primeiro dispositivo...
    int max_perf_device = 0;

    // Define uma performance inicial para posterior comparacao...
    long double max_compute_perf = 0;

    // Looping para verificar dispositivo com melhor performance...
    for( unsigned int index = 0; index < myListSize; ++index )
    {
        // Resgata indice do dispositivo suportado...
        int current_device = _listOfSupportedDevices[ index ];

        // Resgata informacoes do dispositivo...
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties( &deviceProp, current_device );

        // Verifica numero de cores...
        int sm_per_multiproc = 0;
        if( deviceProp.major == 9999 && deviceProp.minor == 9999 )
        {
            sm_per_multiproc = 1;
        }
        else
        {
            sm_per_multiproc = convertSMVer2Cores( deviceProp.major, deviceProp.minor );
        }

        // Computa performance com base no clockRate, numero de processadores
        // e shared memory por multiprocessador...
        long double compute_perf = (long double)( deviceProp.multiProcessorCount ) * deviceProp.clockRate;
                    compute_perf = compute_perf * sm_per_multiproc;

        // Inserindo informacoes de performance do dispositivo...
        std::string msg = "Performance do dispositivo (%s|Index:%d) -> %.2llf Gflops.";
        std::cout << msg << std::endl;
        //long double myDivisor = 1024.0f * 1024.0f * 1024.0f; // for Gflops...
        
        // Registrando mensagem sobre performance...
        /*_execution.logMessageF( msg.c_str(), deviceProp.name, current_device, 
                                (long double)( compute_perf )/( myDivisor ) );*/

        // Verifica se dispositivo atual possui uma
        // melhor performance...
        if( compute_perf > max_compute_perf )
        {
            // If we find GPU with SM major > 2,
            // search only these...
            if( best_SM_arch > 2 )
            {
                // If our device == dest_SM_arch, choose
                // this, or else pass...
                if( deviceProp.major == best_SM_arch )
                {
                    max_compute_perf = compute_perf;
                    max_perf_device = current_device;
                }
            }
            else
            {
                max_compute_perf = compute_perf;
                max_perf_device = current_device;
            }
        }
    }

    // Retorna ID do dispositivo com a melhor performance...
    return max_perf_device;
}


inline int CUDAManager::convertSMVer2Cores( int major, int minor )
{
    // Defines for GPU Architecture types (using the
    // SM version to determine the # of cores per SM)
    typedef struct
    {
        // 0xMm (hexidecimal notation),
        // M = SM Major version, and m = SM minor version.
        int SM;

        // Number of cores.
        int Cores;
    }
    sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10, 8 },  // Tesla Generation (SM 1.0) G80 class
        { 0x11, 8 },  // Tesla Generation (SM 1.1) G8x class
        { 0x12, 8 },  // Tesla Generation (SM 1.2) G9x class
        { 0x13, 8 },  // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
        { 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
        { -1, -1 }
    };

    int index = 0;
    while( nGpuArchCoresPerSM[ index ].SM != -1 )
    {
        if( nGpuArchCoresPerSM[ index ].SM == ( ( major << 4 ) + minor ) )
        {
            return nGpuArchCoresPerSM[ index ].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    fprintf( stderr, "MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n",
             major, minor,
             nGpuArchCoresPerSM[ 7 ].Cores );

    return nGpuArchCoresPerSM[ 7 ].Cores;
}


cudaEvent_t CUDAManager::startClock()
{
    cudaEvent_t start;
    cudaEventCreate( &start );
    cudaEventRecord( start, 0 );

    return start;
}


void CUDAManager::stopClock( cudaEvent_t start, const char* parentFuncion )
{
    cudaEvent_t stop;
    cudaEventCreate( &stop );
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf( "%s : %lf segundos\n", parentFuncion, elapsedTime / 1000 );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
}


float CUDAManager::stopClock( cudaEvent_t start )
{
    cudaEvent_t stop;
    cudaEventCreate( &stop );
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return elapsedTime;
}


bool CUDAManager::aquireGPU()
{
    if( !_validGPUAvailable )
    {
        return false;
    }

    const struct timespec timeout =
    {
        0, ( long int ) 15e7
    };

    int result = pthread_mutex_timedlock( &_mutex, &timeout );
	
    return result == 0;
}


void CUDAManager::releaseGPU()
{
    pthread_mutex_unlock( &_mutex );
}


int CUDAManager::iDivUp( int a, int b )
{
    return ( a % b != 0 ) ? ( a / b + 1 ) : ( a / b );
}


int CUDAManager::getCurrenDeviceIndex() const
{
    return _currentDeviceIndex;
}


CUDAManager::CUDA_MANAGER_STATE CUDAManager::getState() const
{
    return _state;
}
