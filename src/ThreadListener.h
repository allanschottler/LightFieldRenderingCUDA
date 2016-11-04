#ifndef THREADLISTENER_H
#define	THREADLISTENER_H

class Thread;

class ThreadListener 
{
    public:
        
        // classe que define status da thread            
        enum ThreadState
        {
            THREAD_CANCELED, // thread foi cancelada
            THREAD_ABORT, // thread abortou, erro de execucao
            THREAD_CONCLUDED, // thread conclui corretamente
            THREAD_NOTIFICATION // thread deseja notificar listener no meio da execucao
        };
        
        // destrutor virtual
        virtual ~ThreadListener() {};

        
        /**
         * Recebe status de uma thread
         * @param[in] thread Thread associada ao estado
         * @param[in] state Estado da thread*/
        virtual void receiveThreadState( Thread* thread, const ThreadState& state ) = 0;


};

#endif	/* THREADLISTENER_H */

