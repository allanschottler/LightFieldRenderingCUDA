#include <list>

#include "Thread.h"

Thread::Thread() : WorkerThread(),
    _description( "Thread" )
{
    _isCanceled = false;
    _percentage = 0;
    _isFinished = false;
    _hasListenerNotification = false;
}

Thread::~Thread()
{
}

bool Thread::run()
{
    if( _isFinished )
        return false;
    
    if( _isCanceled )
        return false;
    
    executeStep();
    
    return true;
}

void Thread::finish()
{
    _isFinished = true;
    
    stop();
}


void Thread::cancel()
{
    _isCanceled = true;
    
    join();
    
    cancelCleanUp();
}


void Thread::cancelLite()
{
    _isCanceled = true;
    setDescription( "Cancelando processo..." );
}


void Thread::cancelCleanUp()
{
    // implementado por classe especifica
}

bool Thread::isCanceled() const
{
    return _isCanceled;
}

float Thread::getPercentage() const
{
    return _percentage;
}

void Thread::setPercentage( float percentage )
{
    _percentage = percentage;
}

bool Thread::hasListenerNotification() const
{
    return _hasListenerNotification;
}

void Thread::notifyListener()
{
    _hasListenerNotification = true;
}

void Thread::clearListenerNotification()
{
    _hasListenerNotification = false;
}

const std::string Thread::getDescription()
{
    // mutex para impedir que thread esteja mudando descricao enquanto alguem está a consultando assincronamente
    _changeDescriptionMutex.lock();
    std::string description = _description.c_str(); // força copia da memoria
    _changeDescriptionMutex.unlock();
    
    return description;
}



void Thread::setDescription( std::string description )
{
    _changeDescriptionMutex.lock();
    
    _description = description.c_str(); // força copia da memoria
    
    _changeDescriptionMutex.unlock();
}


