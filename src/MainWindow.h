/* 
 * File:   MainWindow.h
 * Author: allan
 *
 * Created on April 12, 2016, 11:49 PM
 */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <string>

class MainWindow 
{
public:
    
    MainWindow( std::string title );
    
    virtual ~MainWindow();
    
    
private:
        
    std::string _title;
     
};

#endif /* MAINWINDOW_H */

