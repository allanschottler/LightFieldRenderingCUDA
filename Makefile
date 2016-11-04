# ------------------------------------------------
# Generic Makefile
#
# Author: yanick.rochon@gmail.com
# Date  : 2011-08-10
#
# Changelog :
#   2010-11-05 - first version
#   2011-08-10 - added structure : sources, objects, binaries
#                thanks to http://stackoverflow.com/users/128940/beta
# ------------------------------------------------

# project name (generate executable with this name)
TARGET   = lfr

CC       = g++	
NVCC     = /local/allanws/v3o2/dependencies/ext/cuda/bin/nvcc
# compiling flags here
CFLAGS   = -Wall -g -I. -std=c++11 -D_GLIBCXX_PARALLEL
NVFLAGS  = -lGL -lGLU -lglut -lpthread -lGLEW -lcudart -arch=sm_20 

LINKER   = g++ -o
# linking flags here
LFLAGS   = -Wall -g -I. -lm

#GTKFLAGS = -export-dynamic `pkg-config --cflags --libs gtk+-3.0 gtkglext-1.0 gtkglext-x11-1.0`

# change these to set the proper directories where each files should be
SRCDIR   = src
OBJDIR   = obj
BINDIR   = bin

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.h)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
	
NVSOURCES  := $(wildcard $(SRCDIR)/*.cu)
NVOBJECTS:= $(NVSOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.cu.o)
rm       = rm -f

GTKINCLUDE  = -I/home/p/libs/libsgtk_64/include/gtkglext-1.0 -I/home/p/libs/libsgtk_64/lib/gtkglext-1.0/include -I/home/p/libs/libsgtk_64/include/gtk-2.0 -I/home/p/libs/libsgtk_64/lib/gtk-2.0/include -I/home/p/libs/libsgtk_64/include/pango-1.0 -I/home/p/libs/libsgtk_64/include/gio-unix-2.0/ -I/home/p/libs/libsgtk_64/include -I/home/p/libs/libsgtk_64/include/gdk-pixbuf-2.0 -I/home/p/libs/libsgtk_64/include/cairo -I/home/p/libs/libsgtk_64/include/glib-2.0 -I/home/p/libs/libsgtk_64/lib/glib-2.0/include -I/home/p/libs/libsgtk_64/include/pixman-1 -I/home/p/libs/libsgtk_64/include/freetype2 -I/home/p/libs/libsgtk_64/include/libpng15 -I/home/p/libs/libsgtk_64/include/atk-1.0
GLINCLUDE   = -I/home/p/libs/freeglut/2.6.0/include -I/home/v/allanws/v3o2/dependencies/ext/glew/include
OSGINCLUDE  = -I/local/allanws/v3o2/dependencies/ext/OSG/include  
CUDAINCLUDE = -I/local/allanws/v3o2/dependencies/ext/cuda_Linux64e6/include
OCVINCLUDE  = -I/home/p/libs/OpenCV-2.4.0/include/opencv

INCLUDE = $(GTKINCLUDE) $(OSGINCLUDE) $(GLINCLUDE) $(CUDAINCLUDE) $(OCVINCLUDE)

GTKLIBS = -L/home/p/libs/libsgtk_64/lib -L/usr/lib64 -lgtkglext-x11-1.0 -lgdkglext-x11-1.0 -lGLU -lGL -lXmu -lXt -lXxf86vm -lXext -lX11 -lgtk-x11-2.0 -lpangox-1.0 -lgdk-x11-2.0 -lXi -lXinerama -lXext -latk-1.0 -lpangoft2-1.0 -lpangocairo-1.0 -lgdk_pixbuf-2.0 -lSM -lICE -lX11 -lgio-2.0 -lcairo -lpango-1.0 -lfreetype -lfontconfig -lgobject-2.0 -lgmodule-2.0 -lrt -lglib-2.0
OSGLIBS = -L/local/allanws/v3o2/dependencies/ext/OSG/lib/Linux64e6 -losg -losgUtil -losgGA -losgDB -losgText -losgViewer -losgSim -lOpenThreads -fopenmp
GLLIB   = -L/home/p/libs/freeglut/2.6.0/lib/Linux64e6 -L/home/v/allanws/v3o2/dependencies/ext/glew/lib/Linux64e6 -lGL -lGLU -lglut -lGLEW -lpthread
CUDALIB = -L/local/allanws/v3o2/dependencies/ext/cuda_Linux64e6/lib64 -lcudart
OCVLIB  = -L/home/p/libs/OpenCV-2.4.0/lib -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_ts -lopencv_video -lopencv_videostab

LIBSDIR = $(GTKLIBS) $(OSGLIBS) $(GLLIB) $(CUDALIB) $(OCVLIB)

$(BINDIR)/$(TARGET) : $(OBJECTS) $(NVOBJECTS)
	@$(LINKER) $@ $(LFLAGS) $(INCLUDE) $(NVOBJECTS) $(OBJECTS) $(LIBSDIR) 
	@echo "Linking complete!"

$(NVOBJECTS) : $(OBJDIR)/%.cu.o : $(SRCDIR)/%.cu 
	@$(NVCC) $(NVFLAGS) -c $< -o $@
	@echo "Compiled (NVCC) "$<" successfully!"
	
$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@
	@echo "Compiled "$<" successfully!"	

.PHONEY: clean
clean:
	@$(rm) $(OBJECTS) $(NVOBJECTS)
	@echo "Cleanup complete!"

.PHONEY: remove
remove: clean
	@$(rm) $(BINDIR)/$(TARGET)
	@echo "Executable removed!"
