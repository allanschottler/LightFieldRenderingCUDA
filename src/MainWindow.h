/* 
 * File:   MainWindow.h
 * Author: allan
 */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <string>
#include <gtk/gtk.h>

#include "osggtkdrawingarea.h"

class MainWindow 
{
public:
    
    MainWindow( std::string title );
    
    virtual ~MainWindow();
    
    void show() { gtk_widget_show_all( GTK_WIDGET( _dialog ) ); };
    
    OSGGTKDrawingArea& getCanvas() { return _canvas; };
    
    void setFocalPlaneSpinAdjustment( double minValue, double maxValue, double value );
    
private:

    // CALLBACKS
    static gboolean onDestroy();
    static gboolean onIdle( gpointer pointer );
    static gboolean onOpenButtonClicked( GtkWidget* button, gpointer pointer );
    static gboolean onQuitButtonClicked( GtkWidget* button, gpointer pointer );
    static gboolean onAboutButtonClicked( GtkWidget* button, gpointer pointer );
    static gboolean onFocalPlaneChanged( GtkWidget* spinbutton, gpointer pointer );
    
    std::string _title;
    
    OSGGTKDrawingArea _canvas;
    
    //Dialogs
    GtkWidget* _dialog;
    GtkWidget* _aboutDialog;
    
    //Menu
    GtkWidget* _openButton;
    GtkWidget* _quitButton;
    GtkWidget* _aboutButton;
    
    // Sidebar    
    GtkWidget* _focalPlaneSpinButton;  
};

#endif /* MAINWINDOW_H */

