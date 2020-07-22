#ifndef VECTOR3_HPP_INCLUDED
#define VECTOR3_HPP_INCLUDED

#include "utils.cuh"

using namespace std;

class vector3_t {
public:
	float x, y, z;

	__hybrid__ vector3_t() {}
	__hybrid__ vector3_t(const float& x, const float& y, const float& z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}
	__hybrid__ ~vector3_t() {}

	__hybrid__ vector3_t operator+(const vector3_t& vector) {
		return vector3_t(this->x + vector.x, this->y + vector.y, this->z + vector.z);
	}

	__hybrid__ vector3_t operator-(const vector3_t& vector) {
		return vector3_t(this->x - vector.x, this->y - vector.y, this->z - vector.z);
	}

	__hybrid__ vector3_t operator*(const float& scalar) {
		return vector3_t(this->x * scalar, this->y * scalar, this->z * scalar);
	}

	__hybrid__ bool operator==(const vector3_t& vector) {
		return fabsf(this->x - vector.x) < EPS && fabsf(this->y - vector.y) < EPS && fabsf(this->z - vector.z) < EPS;
	}

	__hybrid__ float dot(const vector3_t& vector) {
		return this->x*vector.x + this->y*vector.y + this->z*vector.z;
	}

	__hybrid__ vector3_t cross(const vector3_t& vector) {
		return vector3_t(this->y*vector.z - this->z*vector.y, this->z*vector.x - this->x*vector.z, this->x*vector.y - this->y*vector.x);
	}

	__hybrid__ float norm2() {
		return this->x*this->x + this->y*this->y + this->z*this->z;
	}

	__hybrid__ float norm() {
		return sqrt(this->x*this->x + this->y*this->y + this->z*this->z);
	}

	__hybrid__ void normalize() {
		float aux = this->x*this->x + this->y*this->y + this->z*this->z;

		if (aux != 0.0 && aux != 1.0)
			aux = 1.0 / sqrt(aux);

		this->x *= aux;
		this->y *= aux;
		this->z *= aux;
	}

	__hybrid__ vector3_t normalized() {
		float aux = this->x*this->x + this->y*this->y + this->z*this->z;

		if (aux != 0.0 && aux != 1.0)
			aux = 1.0 / sqrt(aux);

		return vector3_t(this->x * aux, this->y * aux, this->z * aux);
	}

	friend ostream& operator<<(ostream& os, const vector3_t& vector);
};

ostream& operator<<(ostream& os, const vector3_t& vector) {
	os << "(" << vector.x << ", " << vector.y << ", " << vector.z << ")";
	return os;
}

#endif // VECTOR3_HPP_INCLUDED