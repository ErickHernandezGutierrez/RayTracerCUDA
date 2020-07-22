#ifndef INTERDATA_HPP_INCLUDED
#define INTERDATA_HPP_INCLUDED

#include "utils.cuh"
#include "colour.cuh"
#include "vector3.cuh"
#include "point3.cuh"
#include "material.cuh"

class interdata_t {
public:
	point3_t interpoint;
	vector3_t normal;
	material_t material;
	//float u, v;

	__hybrid__ interdata_t() {}
	__hybrid__ interdata_t(point3_t interpoint, vector3_t normal, material_t material) {
		this->interpoint = interpoint;
		this->normal = normal;
		this->material = material;
	}
	__hybrid__ ~interdata_t() {}
};

#endif // INTERDATA_HPP_INCLUDED