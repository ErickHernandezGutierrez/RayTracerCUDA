#ifndef SPHERE_HPP_INCLUDED
#define SPHERE_HPP_INCLUDED

/*#include <cmath>
#include "primitive.hpp"
#include "material.hpp"//*/
#include "utils.cuh"
#include "vector3.cuh"
#include "point3.cuh"
#include "ray.cuh"
#include "material.cuh"
#include "interdata.cuh"

using namespace std;

class sphere_t {
public:
	point3_t center;
	material_t material;
	float radius, radiusSquared, radiusInversed;

	__hybrid__ sphere_t() {}
	sphere_t(material_t material, point3_t center, float radius){
		this->material = material;
		this->center = center;
		this->radius = radius;
		this->radiusSquared = radius*radius;
		this->radiusInversed = 1.0f / radius;
	}
	__hybrid__ sphere_t(point3_t center, float radius) {
		this->center = center;
		this->radius = radius;
		this->radiusSquared = radius*radius;
		this->radiusInversed = 1.0f / radius;
	}
	__hybrid__ ~sphere_t() {}

	__hybrid__ float raycast(const ray_t& ray) {
		point3_t  o = ray.origin;
		vector3_t d = ray.direction;

		vector3_t l = center - o;
		float s = l.dot(d);
		float ll = l.dot(l);

		if (s < 0.0f && ll > radiusSquared)
			return -1.0f; //no intersection

		float mm = ll - s*s;

		if (mm > radiusSquared)
			return -1.0f; //no intersection

		float q = sqrt(radiusSquared - mm);
		float t = ll > radiusSquared ? s - q : s + q;

		return t; //intersection
	}

	
	__hybrid__ interdata_t interdata(const ray_t& ray, const float& t) {
		point3_t  o = ray.origin;
		vector3_t d = ray.direction;

		point3_t interpoint = o + (d*t);
		vector3_t normal = (interpoint - center)*(-1.0);

		return interdata_t(interpoint, normal.normalized(), material);
	}

	/*virtual bool shadowIntersect(ray_t shadow_ray) {
		return false;
	}//*/
};

#endif // SPHERE_HPP_INCLUDED