#ifndef RAY_HPP_INCLUDED
#define RAY_HPP_INCLUDED

#include "utils.cuh"
#include "point3.cuh"
#include "vector3.cuh"

class ray_t {
public:
	point3_t origin;
	vector3_t direction;

	__hybrid__ ray_t() {}
	__hybrid__ ray_t(point3_t origin, vector3_t direction) {
		this->origin = origin;
		this->direction = direction;
	}
	__hybrid__ ~ray_t() {}

	friend ostream& operator<<(ostream& os, const ray_t& ray);
};

ostream& operator<<(ostream& os, const ray_t& ray) {
	os << "origin:" << ray.origin << " " << "direction: " << ray.direction;
	return os;
}

__hybrid__ ray_t getReflectedRay(point3_t& P, vector3_t& I, vector3_t& N) {
	vector3_t direction = (I - (N*(2.0f * I.dot(N)))).normalized();
	point3_t origin = P + (direction * EPS);
	return ray_t(origin, direction);
}

__hybrid__ ray_t getImageRay(const int& pixel_posy, const int& pixel_posx, const int& width, const int& height, const float& ratio, const float& fov) {
	float px = (2.0 * ((pixel_posx + 0.5) / width) - 1.0) * tan(fov / 2.0f * PI / 180.0f) * ratio;
	float py = (1.0 - 2.0 * ((pixel_posy + 0.5) / height)) * tan(fov / 2.0f * PI / 180.0f);

	point3_t  origin(0.0f, 0.0f, 0.0f);
	vector3_t direction(px, py, -1.0f);
	direction.normalize();

	return ray_t(origin, direction);
}

__global__ void generateImageRays(ray_t* image_rays, int width, int height, float ratio, float fov) {
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if (i >= height || j >= width) return;

	image_rays[i*width + j] = getImageRay(i, j, width, height, ratio, fov);
}

__global__ void transformImageRays(ray_t* image_rays, ray_t* scene_rays, mat4x4_t LAM, int width, int height) {
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if (i >= height || j >= width) return;

	/*scene_rays[i*width + j].origin = LAM * image_rays[i*width + j].origin;
	scene_rays[i*width + j].direction = LAM * image_rays[i*width + j].direction;//*/

	image_rays[i*width + j].origin    = LAM * image_rays[i*width + j].origin;
	image_rays[i*width + j].direction = LAM * image_rays[i*width + j].direction;
}

#endif // RAY_HPP_INCLUDED
