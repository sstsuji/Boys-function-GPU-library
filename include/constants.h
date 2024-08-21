#ifndef CONSTANTS_H
#define CONSTANTS_H


#include <cstdint>

using Degree = uint8_t;
using Sortkey = uint8_t;



#ifndef LUT_XI_MAX
#define LUT_XI_MAX 32.0
#endif

#ifndef LUT_XI_INTERVAL
#define LUT_XI_INTERVAL 0.03125
#endif

#ifndef LUT_NUM_XI
#define LUT_NUM_XI 1024
#endif

#ifndef LUT_K_MAX
#define LUT_K_MAX 5
#endif

#define A_TR 0.352905920120321
#define B_TR 0.015532762923351
#define A_RS 0.064048916778075
#define B_RS 28.487431543672
#define A 0.03768724  
#define B 0.60549623
#define C 6.32743473
#define D 10.350421

//#define FULLMASK 0xffffffff


#endif // CONSTANTS_H