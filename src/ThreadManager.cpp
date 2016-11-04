#include <map>
#include <vector>

#include "ThreadManager.h"
#include "Thread.h"
#include <string>
#include <typeinfo>
#include <list>
#include <iostream>

ThreadManager* ThreadManager::_instance = 0;

ThreadManager::ThreadManager() 
{   
}

ThreadManager::~ThreadManager() 
{
    clear();
}

void ThreadManager::clear()
{
    ThreadListenerMap::iterator iThread;
    
    // TODO: destruir janelas
    for( iThread = _threadListenerMap.begin(); iThread != _threadListenerMap.end(); ++iThread )
    {
        Thread* thread = (*iThread).first;
        ThreadListener* listener = (*iThread).second;
        
        if( thread->isRunning() )
        {
            thread->cancel();
            
            // avisa listener do estado da thread
            if( listener )
            {
                // thread parou, checar se foi cancelada ou abortada
                ThreadListener::ThreadState state = ThreadListener::THREAD_CANCELED;
                listener->receiveThreadState( thread, state );
            }
        }
        else
        {
            // thread parou, checar se foi cancelada ou abortada
            ThreadListener::ThreadState state = ThreadListener::THREAD_CONCLUDED;
            
            if( thread->hadExecutionError() ) 
            {
                state = ThreadListener::THREAD_ABORT;
            }
            else if( thread->isCanceled() ) 
            {
                state = ThreadListener::THREAD_CANCELED;
            }
            
            // avisa listener do estado da thread
            if( listener )
               listener->receiveThreadState( thread, state );
        }
        
        delete thread;
    }
    
    _threadListenerMap.clear();
}

ThreadManager* ThreadManager::getInstance()
{
    if( _instance == 0 )
    {
        _instance = new ThreadManager();
    }
    
    return _instance;
}

void ThreadManager::destroy()
{
    delete _instance;
    _instance = 0;
}


void ThreadManager::startThread( ThreadListener* threadListener, Thread* thread )
{
    _threadListenerMap[thread] = threadListener;
    
    thread->start();
}


void ThreadManager::checkThreads()
{
    ThreadListenerMap::iterator iThread;
    
    std::vector< ThreadListenerMap::iterator > selectedIterators;
    
    selectedIterators.reserve( _threadListenerMap.size() ); 
    
    for( iThread = _threadListenerMap.begin(); iThread != _threadListenerMap.end(); ++iThread )
    {
        Thread* thread = (*iThread).first;
        ThreadListener* listener = (*iThread).second;
        
        if( thread->isRunning() )
        {
            // verifica se deve notificar listener
            if( thread->hasListenerNotification() && listener )
            {
                thread->clearListenerNotification();
                listener->receiveThreadState( thread, ThreadListener::THREAD_NOTIFICATION );
            }
        }
        // se thread estiver parada
        else
        {
            // thread parou, checar se foi cancelada ou abortada
            ThreadListener::ThreadState state = ThreadListener::THREAD_CONCLUDED;
            
            if( thread->hadExecutionError() ) 
            {
                state = ThreadListener::THREAD_ABORT;
                
                std::cout << "THREAD ABORT\n";
            }
            else if( thread->isCanceled() ) 
            {
                state = ThreadListener::THREAD_CANCELED;
                
                std::cout << "THREAD CANCELED\n";
            }
            
            selectedIterators.push_back( iThread );
            
            // avisa listener do estado da thread
            if( listener )
                listener->receiveThreadState( thread, state );
        }
    }
    
    // apos todos os listeneres terem sido avisados sobre as threads, deleta threads terminadas e retira do mapa
    for( unsigned int i = 0; i < selectedIterators.size(); i++ )
    {
        iThread = selectedIterators[i];
        Thread* thread = (*iThread).first;
        
        // remove par thread/listener do mapa de listener
        _threadListenerMap.erase( thread );
        
        // deleta thread
        delete thread;
    }
}

bool ThreadManager::hasThreads() const
{
    return !_threadListenerMap.empty();
}


void ThreadManager::removeListener( Thread* thread )
{
    if( _threadListenerMap.count( thread ) != 0 )
        _threadListenerMap[thread] = 0;
}



