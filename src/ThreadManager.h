#ifndef THREADMANAGER_H
#define	THREADMANAGER_H

#include "ThreadListener.h"
//#include "ProgressBar.h"
//#include "logger/vlog.h"
#include <map>

class ThreadManager 
{
    public:
        
        /**
         * Funcao do singleton para pegar instancia
         */
        static ThreadManager* getInstance();
        
        /**
         * Funcao para destruir instancia do singleton
         */
        static void destroy();
    
        /**
         * Destrutor
         */
        virtual ~ThreadManager();
        
        /**
         * Checa status das threads e avisa listeners
         */
        void checkThreads();
        
        /**
         * Roda uma thread
         * @param threadListener listener da trhead
         * @param thread thread a ser disparada
         * @param[in] parameters Parametros do progresso
         */
        void startThread( ThreadListener* threadListener, Thread* thread );
        
        /**
         * retorna se existem threads registradas no momento
         * @return 
         */
        bool hasThreads() const;
        
        /**
         * Faz thread parametro nao estar mais associada a nenhum listener
         */
        void removeListener( Thread* thread );
        
    private:
        
        /**
         * Construtor privado, singleton
         */
        ThreadManager();
        
        /**
         * clear nas threads, cancela todas e as deleta
         */
        void clear();
        
        // instancia singleton
        static ThreadManager* _instance;
        
    protected:
        // tipo mapa de thread
        typedef std::map< Thread*, ThreadListener* > ThreadListenerMap;
        
        //typedef std::map< Thread*, ProgressBar* > ThreadProgressBarMap;
        
        //typedef std::map< Thread*, vlog::CommandExecution* > ThreadLoggerExecutionMap;
        
        // mapa de threads
        ThreadListenerMap _threadListenerMap;
        
        // mapa de progressos
        //ThreadProgressBarMap _threadProgressBarMap;
        
        // mapa de execucoes do logger
        //ThreadLoggerExecutionMap _threadLoggerExecutionMap;
};

#endif	/* THREADMANAGER_H */

