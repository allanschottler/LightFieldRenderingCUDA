
#include "LightFieldApplication.h"

int main(int argc, char **argv)
{    
    gtk_init( &argc, &argv );
    gtk_gl_init( &argc, &argv );

    LightFieldApplication::getInstance();
    
    gtk_main();
        
    return 0;
}
