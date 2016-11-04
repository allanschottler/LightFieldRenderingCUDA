#ifndef THREAD_H
#define	THREAD_H

#include "WorkerThread.h"
#include <mutex>
#include <list>
#include <string>

class Thread : public WorkerThread
{
    public:
        
        /**
         * Construtor
         */
        Thread();
    
        /**
         * Destrutor
         */
        virtual ~Thread();
        
        /**
         * Funcao de execucao da thread a ser implementada
         * @return 
         */
        bool run();
        
        
        /**
         * Cancela thread
         */
        virtual void cancel();
        
        /**
         * Cancela thread sem join e cancelCleanup
         */
        void cancelLite();
        
        /**
         * Retorna se foi cancelada
         */
        bool isCanceled() const;
        
        /**
         * Retorna percentagem
         */
        float getPercentage() const;
        
        /**
         * Define percentagem
         * @param percentage
         */
        void setPercentage( float percentage );
        
        /**
         * Retorna se manager deve notificar listener
         * @return 
         */
        bool hasListenerNotification() const;
        
        /**
         * Limpa infomacao de se há notificacoes para o listener
         */
        void clearListenerNotification();
        
        /**
         * Retorna descrição do passo atual da thread
         */
        const std::string getDescription();

        /**
         * Define a descrição da thread
         */
        void setDescription( std::string description );
        
        /**
         * Marca que listener deve ser avisado pelo manager
         */
        void notifyListener();

    protected:
        
        /**
         * Finaliza thread, deve ser chamada pela implementacao filha ao termino da thread
         */
        void finish();
        
        /**
         * Funcao que limpa thread apos ser cancelada
         */
        virtual void cancelCleanUp();
        
        /**
         * Executa um passo da thread
         */
        virtual void executeStep() = 0;
        
        // define se foi cancelada
        bool _isCanceled;
        
        // percentagem da thread
        float _percentage;
        
        // define a descrição do passo atual da thread
        std::string _description;

    private:
        
        // define se thread terminou
        bool _isFinished;
        
        // define se deve avisar listener
        bool _hasListenerNotification;
        
        std::mutex _changeDescriptionMutex;
};

#endif	/* THREAD_H */

