/* 
 * File:   PNGLoaderThread.h
 * Author: allan
 */

#ifndef PNGLOADERTHREAD_H
#define	PNGLOADERTHREAD_H

#include "Thread.h"

#include <vector>

class PNGLoaderThread : public Thread
{
public:
    
    PNGLoaderThread( std::string pngFilePath, int threadID );
    
    virtual ~PNGLoaderThread();
    
    void executeStep();
    
    void cancelCleanUp();
    
    std::vector< unsigned char > getLoadedImage() { return _loadedImage; };
    
    int getThreadID() { return _threadID; };
    
private:

    std::string _pngFilePath;
    
    int _threadID;
    
    std::vector< unsigned char > _loadedImage;
};

#endif	/* PNGLOADERTHREAD_H */

