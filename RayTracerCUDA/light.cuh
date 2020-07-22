#ifndef LIGHT_HPP_INCLUDED
#define LIGHT_HPP_INCLUDED

#include "utils.cuh"
#include "colour.cuh"
#include "vector3.cuh"
#include "point3.cuh"

class light_t {
public:
	point3_t position;
	colour_t color;
	float intensity;
	colour_t irradiance;

	__hybrid__ light_t() {}
	__hybrid__ light_t(point3_t position, colour_t color, float intensity) {
		this->position = position;
		this->color = color;
		this->intensity = intensity;
		this->irradiance = color * intensity;
	}
	__hybrid__ ~light_t() {}
};

#endif // LIGHT_HPP_INCLUDED