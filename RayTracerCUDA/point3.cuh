#ifndef POINT3_HPP_INCLUDED
#define POINT3_HPP_INCLUDED

#include "utils.cuh"
#include "vector3.cuh"

using namespace std;

class point3_t {
public:
	float x, y, z;

	__hybrid__ point3_t() {}
	__hybrid__ point3_t(float x, float y, float z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}
	__hybrid__ ~point3_t() {}

	__hybrid__ vector3_t operator-(const point3_t& point) {
		return vector3_t(this->x - point.x, this->y - point.y, this->z - point.z);
	}

	__hybrid__ point3_t operator+(const vector3_t& vector) {
		return point3_t(this->x + vector.x, this->y + vector.y, this->z + vector.z);
	}

	__hybrid__ float dot(const vector3_t& vector) {
		return this->x*vector.x + this->y*vector.y + this->z*vector.z;
	}

	__hybrid__ float dot(const point3_t& point) {
		return this->x*point.x + this->y*point.y + this->z*point.z;
	}

	friend ostream& operator<<(ostream& os, const point3_t& point);
};

ostream& operator<<(ostream& os, const point3_t& point) {
	os << "(" << point.x << ", " << point.y << ", " << point.z << ")";
	return os;
}

#endif // POINT3_HPP_INCLUDED
