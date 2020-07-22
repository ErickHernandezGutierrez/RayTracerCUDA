#ifndef MATERIAL_HPP_INCLUDED
#define MATERIAL_HPP_INCLUDED

#include "utils.cuh"
#include "colour.cuh"

class material_t {
public:
	colour_t kd;
	colour_t ks;
	float sh;
	float reflective;

	__hybrid__ material_t() {}
	__hybrid__ material_t(colour_t kd, colour_t ks, float sh) {
		this->kd = kd;
		this->ks = ks;
		this->sh = sh;
		this->reflective = false;
	}
	__hybrid__ material_t(colour_t kd, colour_t ks, float sh, bool reflective) {
		this->kd = kd;
		this->ks = ks;
		this->sh = sh;
		this->reflective = reflective;
	}
	__hybrid__ ~material_t() {}
};

#endif // MATERIAL_HPP_INCLUDED