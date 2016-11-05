/* 
 * File:   MainWindow.cpp
 * Author: allan
 */

#include "MainWindow.h"
#include "LightFieldApplication.h"
#include "ThreadManager.h"

MainWindow::MainWindow( std::string title ) :
    _title( title ) 
{
    GError* error = NULL;
    GtkBuilder* builder = gtk_builder_new();
    
    std::string gladePath( g_get_current_dir() );
    gladePath.append( "/../data/glade/MainWindow.glade" );
    
    if( !gtk_builder_add_from_file( builder, gladePath.c_str(), &error ) )
    {
        g_warning( "Nao foi possivel abrir o arquivo: %s", error->message );
        g_error_free( error );
    }
    
    _dialog = GTK_WIDGET( gtk_builder_get_object( builder, "window1" ) );
    
    gtk_window_set_title( GTK_WINDOW( _dialog ), _title.c_str() );
    
    GtkWidget* canvasBox = GTK_WIDGET( gtk_builder_get_object( builder, "alignmentCanvas" ) );
    
    if( _canvas.createWidget( 800, 800 ) ) 
    {
        gtk_box_pack_start( GTK_BOX( canvasBox ), _canvas.getWidget(), true, true, 2 );        
    }
    
    _aboutDialog = GTK_WIDGET( gtk_builder_get_object( builder, "aboutdialog1" ) );  
    
    _quitButton = GTK_WIDGET( gtk_builder_get_object( builder, "imagemenuitem5" ) );
    _aboutButton = GTK_WIDGET( gtk_builder_get_object( builder, "imagemenuitem10" ) );
        
    g_timeout_add( 15, (GSourceFunc)( &MainWindow::onIdle ), this );
    
    g_signal_connect( G_OBJECT( _dialog ), "destroy", G_CALLBACK( &MainWindow::onDestroy ), NULL );
    g_signal_connect( G_OBJECT( _dialog ), "delete_event", G_CALLBACK( &MainWindow::onDestroy ), NULL );    
    
    g_signal_connect( G_OBJECT( _quitButton ), "activate", G_CALLBACK( &MainWindow::onQuitButtonClicked ), _dialog );
    g_signal_connect( G_OBJECT( _aboutButton ), "activate", G_CALLBACK( &MainWindow::onAboutButtonClicked ), _dialog );
        
    g_object_set_data( ( GObject* ) _dialog, "THIS", ( gpointer )this );
}


MainWindow::~MainWindow() 
{
}


gboolean MainWindow::onDestroy()
{
    gtk_main_quit();
    
    return FALSE;
}

gboolean MainWindow::onIdle( gpointer pointer )
{
    MainWindow* dialog = reinterpret_cast< MainWindow* >( pointer );
        
    if( ThreadManager::getInstance()->hasThreads() )
        ThreadManager::getInstance()->checkThreads();
    
    dialog->_canvas.queueDraw();
    
    return TRUE;
}

gboolean MainWindow::onQuitButtonClicked( GtkWidget* button, gpointer pointer )
{
    return MainWindow::onDestroy();
}

gboolean MainWindow::onAboutButtonClicked( GtkWidget* button, gpointer pointer )
{
    gpointer result = g_object_get_data( ( GObject* ) pointer, "THIS" );
    
    if( result == NULL )
        return FALSE;
    
    MainWindow* dialog = reinterpret_cast< MainWindow* >( result );
    
    std::string aboutString = 
        "Trabalho sobre Light Field Rendering.\n"
        "Prof. Waldemar Celes\n"
        "\n"
        "LEFT MOUSE = Rotação\n"
        "RIGHT MOUSE = Zoom\n"
        "MIDDLE MOUSE = Pick/Pan\n"
        "SPACE = Camera inicial\n";
            
    gtk_widget_destroy( dialog->_aboutDialog );
    
    dialog->_aboutDialog = gtk_about_dialog_new();
    gtk_about_dialog_set_name( GTK_ABOUT_DIALOG( dialog->_aboutDialog ), "Light Field Rendering" );
    gtk_about_dialog_set_version( GTK_ABOUT_DIALOG( dialog->_aboutDialog ), "v1.0" );
    gtk_about_dialog_set_comments( GTK_ABOUT_DIALOG( dialog->_aboutDialog ), aboutString.c_str() );
    //gtk_about_dialog_set_authors( GTK_ABOUT_DIALOG( dialog->_aboutDialog ), authorName );
    
    gtk_widget_show_all( dialog->_aboutDialog );
    
    return TRUE;
}