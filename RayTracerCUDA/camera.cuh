#ifndef CAMERA_HPP_INCLUDED
#define CAMERA_HPP_INCLUDED

#include "utils.cuh"
#include "point3.cuh"
#include "vector3.cuh"
#include "mat4x4.cuh"

using namespace std;

class camera_t{
public:
	point3_t position;
	vector3_t forward;
	vector3_t right;
	vector3_t up;

	__hybrid__ camera_t() {}
	__hybrid__ camera_t(point3_t position, vector3_t forward, vector3_t right, vector3_t up) {
		this->position = position;
		this->forward = forward;
		this->right = right;
		this->up = up;
	}
	__hybrid__ ~camera_t() {}

	mat4x4_t getLookAtMatrix() {
		return mat4x4_t(right.x, up.x, forward.x, position.x,
			right.y, up.y, forward.y, position.y,
			right.z, up.z, forward.z, position.z,
			0.0f, 0.0f, 0.0f, 1.0f);
	}
};

#endif //CAMERA_HPP_INCLUDED