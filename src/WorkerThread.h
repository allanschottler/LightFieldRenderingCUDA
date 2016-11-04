#ifndef WORKERTHREAD_H
#define	WORKERTHREAD_H

#include <pthread.h>
#include <unistd.h>

class WorkerThread 
{
    public:
    
        /**
         * Construtor
         */
        WorkerThread();

        /**
         * Destrutor
         */
        virtual ~WorkerThread();
        
        /**
         * inicia thread
         * @ret se iniciou com sucesso
         */
        bool start();
        
        /**
         * Para thread
         */
        void stop();
        
        /**
         * Funcao esatica de execucao para passar para o pThread
         * @param instance
         * @return 
         */
        static void *_exec(void *instance);
        
        /**
         * Funcao esatica de execucao serial da thread
         * @param instance
         * @return 
         */
        static void *_execSerial(void *instance);
        
        /**
         * Funcao de execucao da thread a ser implementada
         * @return 
         */
        virtual bool run() = 0;
        
        /**
         * Retorna se thread esta sendo executada
         * @return 
         */
        useconds_t isRunning() const;
        
        /**
         * Da join na thread, fica travado até que thread se conclua
         */
        void join();
        
        /**
         * Retorna se houve erro de execucao
         */
        bool hadExecutionError() const;
    
    protected:

        /**
         * Mata thread
         */
        void cancelExecution();
        
    private:
    
        // referencia para thread da ptrhead
        pthread_t _pThread;
        
        // se ouve erro
        int _error;
        
        // indica se thread esta rodando
        bool _isRunning;
        
        // indica se thread teve erro de execucao
        bool _hadExecutionError;

        // indica se a thread foi cancelada
        bool _hasBeenCancelled;

};

#endif	/* WORKERTHREAD_H */

