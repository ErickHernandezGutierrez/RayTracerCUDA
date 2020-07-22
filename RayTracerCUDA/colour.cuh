#ifndef COLOUR_HPP_INCLUDED
#define COLOUR_HPP_INCLUDED

#include "utils.cuh"

using namespace std;

class colour_t {
public:
	float r, g, b;

	__hybrid__ colour_t() {}
	__hybrid__ colour_t(float r, float g, float b) {
		this->r = r;
		this->g = g;
		this->b = b;
	}
	__hybrid__ ~colour_t() {}

	__hybrid__ colour_t operator+(colour_t color) {
		return colour_t(this->r + color.r, this->g + color.g, this->b + color.b);
	}

	__hybrid__ colour_t operator-(colour_t color) {
		return colour_t(this->r - color.r, this->g - color.g, this->b - color.b);
	}

	__hybrid__ colour_t operator*(float scalar) {
		return colour_t(this->r * scalar, this->g * scalar, this->b * scalar);
	}

	__hybrid__ bool operator==(colour_t color) {
		return fabsf(this->r - color.r) < EPS && fabsf(this->g - color.g) < EPS && fabsf(this->b - color.b) < EPS;
	}

	__hybrid__ void operator+=(const colour_t& color) {
		this->r += color.r;
		this->g += color.g;
		this->b += color.b;
	}

	__hybrid__ void operator-=(const colour_t& color) {
		this->r -= color.r;
		this->g -= color.g;
		this->b -= color.b;
	}

	__hybrid__ void operator*=(const float& scalar) {
		this->r *= scalar;
		this->g *= scalar;
		this->b *= scalar;
	}

	__hybrid__ uchar toInt() {
		uchar R = (uchar)(255.0f * fminf(1.0f, fmaxf(0.0f, r)));
		uchar G = (uchar)(255.0f * fminf(1.0f, fmaxf(0.0f, g)));
		uchar B = (uchar)(255.0f * fminf(1.0f, fmaxf(0.0f, b)));

		return ((B << 16) | (G << 8) | R);
	}

	/*rgb_t get_rgb() {
		rgb_t rgb;

		rgb.red = (unsigned char)(255.0f * fminf(1.0f, fmaxf(0.0f, r)));
		rgb.green = (unsigned char)(255.0f * fminf(1.0f, fmaxf(0.0f, g)));
		rgb.blue = (unsigned char)(255.0f * fminf(1.0f, fmaxf(0.0f, b)));

		return rgb;
	}//*/
};

#endif // COLOUR_HPP_INCLUDED
