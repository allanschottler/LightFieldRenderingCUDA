#ifndef _INT64_H
#define _INT64_H

#include <stdio.h>

#ifdef WIN32
	typedef fpos_t INT64;
#else
	typedef long long INT64;
#endif

#endif

