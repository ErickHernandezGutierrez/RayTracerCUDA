#ifndef UTILS_HPP_INCLUDED
#define UTILS_HPP_INCLUDED

#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define __hybrid__ __host__ __device__

#define EPS 1e-3
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062
#define MAX_DISTANCE 1e6
#define MAX_DEPTH 3

typedef unsigned char uchar;

#endif // UTILS_HPP_INCLUDED