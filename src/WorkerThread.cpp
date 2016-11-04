#include <exception>
#include <cxxabi.h>

#include "WorkerThread.h"

void* WorkerThread::_exec(void *instance) 
{
    int oldtype;
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &oldtype);
            
    WorkerThread *pThis = reinterpret_cast<WorkerThread *>(instance);
    bool exitRet=true;

    try
    {
        // executa thread enquanto _isRunning
        while( pThis->isRunning() && (exitRet = pThis->run()) ) {
                ;
        }
    
    }
    catch( const std::exception& e )
    {
        // ERRO NA THREAD
        pThis->_hadExecutionError = true;
    }
    catch( const abi::__forced_unwind& ) //Caso de cancelamento (pthread_cancel) do centos6
    {
        pThis->_isRunning = false;
        throw;
    }
    catch( ... )
    {
        //Se foi cancelada
        //está no caso de cancelamento (pthread_cancel) do centos5
        if( pThis->_hasBeenCancelled )
        {
            throw;
        }
        
        // ERRO NA THREAD
        pThis->_hadExecutionError = true;
    }

    pThis->_isRunning = false;

    void *ret;
    ret = ( void* )( intptr_t )( exitRet ? 0 : -1 );
    pthread_exit(ret);

    return NULL;
}



void* WorkerThread::_execSerial(void *instance) 
{
    WorkerThread *pThis = reinterpret_cast<WorkerThread *>(instance);
    
    try
    {
        // executa thread enquanto _isRunning
        while( pThis->run() )
        {
                ;
        }
    
    }
    catch(...)
    {
        // ERRO NA THREAD
        pThis->_hadExecutionError = true;
    }
    
    pThis->_isRunning = false;

    return NULL;
}



WorkerThread::WorkerThread() 
{
    _pThread = 0;
    _error = 0;
    _isRunning = false;
    _hadExecutionError = false;
    _hasBeenCancelled = false;
}



WorkerThread::~WorkerThread() 
{
    stop();

    if( _pThread )
        pthread_detach( _pThread );
}



bool WorkerThread::start() 
{
    _isRunning = true;
    _error = pthread_create(&_pThread, NULL, WorkerThread::_exec, (void *)this);
    
    if( _error != 0 ) 
        return false;
    
    return true;
}



void WorkerThread::stop() 
{
    _isRunning = false;
}



useconds_t WorkerThread::isRunning() const
{
    return _isRunning;
}



bool WorkerThread::hadExecutionError() const
{
    return _hadExecutionError;
}



void WorkerThread::join()
{
    if( _pThread )
    {
        pthread_join( _pThread, NULL );
        _pThread = 0;
    }
}



void WorkerThread::cancelExecution()
{
    _hasBeenCancelled = true;

    if( _pThread )
    {
        pthread_cancel( _pThread );
    }

    _isRunning = false;
}


