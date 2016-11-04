#ifndef _FILENAME_H_
#define _FILENAME_H_
#include <string>

/**
 Namespace FileName
 Its methods are used to manipulate a filename.
 */
namespace FileName
{
   /*
   * Appends an extension to the filename if it does not exist.
   *
   * - if the current extension is the same as in param "extension"
   * nothing is done.
   * ("name.ext", "ext") => ("name.ext", "ext")
   *
   * - if the current extension is different of the one in param
   * "extension" the new one is simply concatenated to "filename"
   * ("name.old", "new") => ("name.old.new", "new")
   *
   * - filename must be long enough to hold the new string
   */
   void appendExtension( std::string& filename, std::string extension );

   /*
   * Changes the extension of a file. Any text after the last dot (.)
   * is replaced by the text in param "extension"
   *  - filename must be long enough to hold the new string
   */
   void changeExtension( std::string& filename, std::string extension );

   /*
   * Returns the full name of a file (name + extension)
   * - name must be long enough to hold the full name
   */
   bool getFullName( std::string path, std::string& name );

   /*
   * Returns the full name of a file (directory + name)
   * - name must be long enough to hold the full name
   */
   bool getDirectoryAndName( std::string path, std::string& name );

   /*
   * Returns the name of a file
   * - name must be long enough to hold the name
   * @param[in] path Name of file, including complete path.
   * @param[out] name Name of file, without path and without extension.
   * @return true in case success and false otherwise.
   */
   bool getName( std::string path, std::string& name );

   /*
   * Returns the extension of a file
   * - extension must be long enough to hold the extension
   */
   bool getExtension( std::string , std::string& extension );

   /*
   * Returns the extension of a file converted to upper case
   * - extension must be long enough to hold the extension
   */
   bool getUpperExtension( std::string , std::string& extension );

   /*
   * Returns the extension of a file converted to lower case
   * - extension must be long enough to hold the extension
   */
   bool getLowerExtension( std::string , std::string& extension );


   /*
   * Returns the directory of a file
   * - name must be long enough to hold the directory's name
   */
   bool getDirectory( std::string path, std::string& directory );

   /*
   * Splits the path into two components: full name and directory
   * - The strings are allocated inside the function using 'new'
   * - Non-existing components are returned as NULL pointers
   */
   void split( std::string path, std::string& fullName, std::string& directory );

   /*
   * Splits the path into two components: full name and directory
   * - result strings be long enough to hold the directory's name
   */
   void split( std::string path, std::string& name, std::string& extension, std::string& directory );


   /* Update the slashes to the current system
   */
   void updateSlashes( std::string& path ) ;


   // POINTER FUNCTIONS
   /*
   * Returns the name of a file
   * @return A pointer to the position of path whre filename starts. since it overlaps path's content it should not be deleted
   */
   std::string getName( std::string path );

};


#endif



