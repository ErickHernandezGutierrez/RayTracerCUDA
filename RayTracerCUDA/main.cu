// OPENGL headers
#include "GL\glew.h"
#include "GLFW\glfw3.h"

// CUDA headers
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"

// C++ headers
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>

// RayTracer headers
#include "vector3.cuh"
#include "point3.cuh"
#include "mat4x4.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include "light.cuh"
#include "interdata.cuh"

using namespace std;
typedef unsigned int uint;

// GLFW globals
GLFWwindow* window;
const int WIDTH  = 256;
const int HEIGHT = 256;

// OPENGL globals
GLuint VBO, VAO, EBO;
GLuint shaders_program;
GLuint opengl_texture;
GLfloat vertices[] = {
	//positions         //colors          //texcoords
	 1.0f,  1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, //top right
	 1.0f, -1.0f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, //bottom right
	-1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, //bottom left
	-1.0f,  1.0f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f  //top left
};
GLuint indices[] = {
	0, 1, 3, //first  triangle
	1, 2, 3  //second triangle
};

// CUDA globals
size_t size_texture;
uint num_texels;
uint num_values;
void* cuda_buffer;
cudaGraphicsResource* cuda_texture;
dim3 block(16, 16, 1);
dim3 grid(WIDTH / block.x, HEIGHT / block.y, 1);

// SCENE globals
sphere_t* spheres;
light_t* lights;
__constant__ sphere_t gpu_spheres[3];
__constant__ light_t gpu_lights[3];
const int num_spheres = 3;
const int num_lights = 3;
ray_t* rays;
ray_t* gpu_rays;
ray_t* gpu_scene_rays;
const float RATIO = WIDTH / (float)HEIGHT;
const float FOV = 45.0f;
//uint* buffer;
mat4x4_t LAM;
/*point3_t camera_position;
vector3_t _forward;
vector3_t _up;
vector3_t _right;//*/
float posx = 0.0f;
float posz = 5.0f;
float rot_angle = 10.0f;
//camera_t camera;

// CUDA kernels
__hybrid__ float clamp(float x, float a, float b)
{
	return fmaxf(a, fminf(b, x));
}

__hybrid__ int clamp(int x, int a, int b)
{
	return max(a, min(b, x));
}

__hybrid__ int rgb2int(float r, float g, float b) {
	r = clamp(r, 0.0f, 255.0f);
	g = clamp(g, 0.0f, 255.0f);
	b = clamp(b, 0.0f, 255.0f);
	return (int(b) << 16) | (int(g) << 8) | int(r);
}

__global__ void kernel(uint* buffer, int width) {
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	uchar4 c4 = make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);
	buffer[y*width + x] = rgb2int(c4.z, c4.y, c4.x);
}

void keyboardfunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_A && action == GLFW_PRESS) {
		posx -= 0.1;
		//camera.position.x -= 0.1;
		cout << "A " << posx << endl;
	}
	else if (key == GLFW_KEY_D && action == GLFW_PRESS) {
		posx += 0.1;
		//camera.position.x += 0.1;
	}
	else if (key == GLFW_KEY_W && action == GLFW_PRESS) {
		posz += 0.1;
		//camera.position.z += 0.1;
	}
	else if (key == GLFW_KEY_S && action == GLFW_PRESS) {
		posz -= 0.1;
		//camera.position.z -= 0.1;
	}
}

bool initGLFW() {
	// these hints switch the OpenGL profile to core
	if (!glfwInit()){ return false; }
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // we want OpenGL <= 4.5
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); // we want OpenGL >= 3.3
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // to make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // we do not want the old OpenGL

	// create window object
	window = glfwCreateWindow(WIDTH, WIDTH, "Ray Tracer CUDA", NULL, NULL);
	if (!window) { glfwTerminate(); return false; }

	// initialize GLEW
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	glfwSetKeyCallback(window, keyboardfunc);

	// ensure we can capture ESC from the keyboard
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	return true;
}

bool initGL() {
	// to enforce core profile
	glewExperimental = GL_TRUE;

	// initialize glew
	GLenum err = glewInit();
	glGetError(); // parse first error
	if (err != GLEW_OK) {
		//printf("glewInit failed: %s /n", glewGetErrorString(err));
		cout << "[ERROR] " << glewGetErrorString(err) << endl;
		return false;
	}

	// viewport for x,y to normalized device coordinates transformation
	glViewport(0, 0, WIDTH, HEIGHT);
	
	return true;
}

void initGLShaders(const char* filepath_vertex_shader, const char* filepath_fragment_shader) {
	cout << "[INFO] Loading Shaders:" << endl;

	// create shaders
	GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	
	// read the vertex shader code from file
	string code_vertex_shader;
	ifstream file_vertex_shader(filepath_vertex_shader, std::ios::in);
	if (file_vertex_shader.is_open()) {
		string line = "";
		while (getline(file_vertex_shader, line))
			code_vertex_shader += "\n" + line;
		file_vertex_shader.close();
	}

	// read the fragment shader code from file
	string code_fragment_shader;
	ifstream file_fragment_shader(filepath_fragment_shader, std::ios::in);
	if (file_fragment_shader.is_open()) {
		string line = "";
		while (getline(file_fragment_shader, line))
			code_fragment_shader += "\n" + line;
		file_fragment_shader.close();
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// compile vertex shader
	//printf("Compiling shader: %s\n", vertex_filepath);
	cout << "\t[INFO] Compiling Vertex Shader..." << endl;
	char const * VertexSourcePointer = code_vertex_shader.c_str();
	glShaderSource(vertex_shader, 1, &VertexSourcePointer, NULL);
	glCompileShader(vertex_shader);

	// check vertex shader
	glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(vertex_shader, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		//printf("%s\n", &VertexShaderErrorMessage[0]);
		cout << "\t[ERROR] " << &VertexShaderErrorMessage[0] << endl;
	}

	// compile fragment shader
	//printf("Compiling shader: %s\n", fragment_filepath);
	cout << "\t[INFO] Compiling Fragment Shader..." << endl;
	char const * FragmentSourcePointer = code_fragment_shader.c_str();
	glShaderSource(fragment_shader, 1, &FragmentSourcePointer, NULL);
	glCompileShader(fragment_shader);

	// check fragment shader
	glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(fragment_shader, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		//printf("[ERROR] %s\n", &FragmentShaderErrorMessage[0]);
		cout << "\t[ERROR] " << &FragmentShaderErrorMessage[0] << endl;
	}

	// link the program
	printf("\t[INFO] Linking Program...\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, vertex_shader);
	glAttachShader(ProgramID, fragment_shader);
	glLinkProgram(ProgramID);

	// check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		//printf("[ERROR] %s\n", &ProgramErrorMessage[0]);
		cout << "\t[ERROR] " << &ProgramErrorMessage[0] << endl;
	}

	glDetachShader(ProgramID, vertex_shader);
	glDetachShader(ProgramID, fragment_shader);

	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);

	shaders_program = ProgramID;
	//return ProgramID;//*/
}

void initGLTexture(GLuint* gltexture, cudaGraphicsResource** cutexture, uint texture_width, uint texture_height) {
	
	// create an OpenGL texture
	glGenTextures(1, gltexture); // generate 1 texture
	glBindTexture(GL_TEXTURE_2D, *gltexture); // set it as current target

	// set basic texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// specify 2D texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, texture_width, texture_height, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);

	//register texture with CUDA
	cudaGraphicsGLRegisterImage(cutexture, *gltexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

void initGLBuffers() {
	
	// generate buffers
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	// setup buffers
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute (3 floats)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	// color attribute (3 floats)
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	// texture attribute (2 floats)
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound 
	// vertex buffer object so afterwards we can safely unbind
	glBindVertexArray(0);
}

void initCUDABuffers() {
	num_texels = WIDTH * HEIGHT;
	num_values = num_texels * 4;
	size_texture = sizeof(GLubyte) * num_values;

	cudaMalloc(&cuda_buffer, size_texture);
}

__global__ void trace(ray_t* rays, uint* buffer, int num_actors, int num_luces, int width, int height, colour_t ambient_color) {
	uint j = blockIdx.x*blockDim.x + threadIdx.x;
	uint i = blockIdx.y*blockDim.y + threadIdx.y;
	uint offset = i*width + j;

	if (i >= height || j >= width) return;

	ray_t ray = rays[offset];

	uint nearest_sphere_id = 0;
	float t = 1e7;
	for (int k = 0; k < num_actors; k++) {
		float temp = gpu_spheres[k].raycast(ray);
		if (temp > 0.0 && temp < t) {
			t = temp;
			nearest_sphere_id = k;
		}
	}

	if (t > 0.0 && t < 1e7) {
		interdata_t interdata = gpu_spheres[nearest_sphere_id].interdata(ray, t);
		point3_t interpoint = interdata.interpoint;
		vector3_t normal = interdata.normal;
		material_t material = interdata.material;
		colour_t pixel_color(0.0f, 0.0f, 0.0f);

		pixel_color += ambient_color;

		for (int k = 0; k < num_luces; k++) {
			light_t light = gpu_lights[k];

			vector3_t L = (light.position - interpoint).normalized();
			vector3_t R = ((normal*(2.0f*normal.dot(L))) - L).normalized();
			vector3_t V = (ray.direction * (-1.0f)).normalized();

			colour_t diffuse = light.irradiance * fmaxf(0.0f, normal.dot(L*(-1.0f)));
			colour_t specular = light.irradiance * powf(fmaxf(0.0f, R.dot(V)), material.sh);

			float r = diffuse.r*material.kd.r + specular.r*material.ks.r;
			float g = diffuse.g*material.kd.g + specular.g*material.ks.g;
			float b = diffuse.b*material.kd.b + specular.b*material.ks.b;

			pixel_color += colour_t(r, g, b);
		}

		buffer[i*width + j] = rgb2int(pixel_color.r * 255, pixel_color.g * 255, pixel_color.b * 255);
	}
	else {
		buffer[i*width + j] = rgb2int(100, 100, 100);
	}
}

void generateCUDAImage() {	

	mat4x4_t rot = createRotationMatrix(rot_angle, 'y');
	rot_angle = (rot_angle += 5.0f);
	camera_t camera(rot*point3_t(posx, 0.0f, posz), rot*vector3_t(0.0f, 0.0f, 1.0f), rot*vector3_t(1.0f, 0.0f, 0.0f), rot*vector3_t(0.0f, -1.0f, 0.0f));
	colour_t ambient_color(0.1f, 0.1f, 0.1f);
	LAM = camera.getLookAtMatrix();

	generateImageRays<<<grid, block>>>(gpu_rays, WIDTH, HEIGHT, RATIO, FOV);
	transformImageRays<<<grid, block>>>(gpu_rays, gpu_scene_rays, LAM, WIDTH, HEIGHT);
	trace<<<grid, block>>>(gpu_rays, (uint*)cuda_buffer, num_spheres, num_lights, WIDTH, HEIGHT, ambient_color);

	// ee want to copy cuda_dev_render_buffer data to the texture
	// map buffer objects to get CUDA device pointers
	cudaArray *texture_ptr;
	cudaGraphicsMapResources(1, &cuda_texture, 0);
	cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_texture, 0, 0);

	cudaMemcpyToArray(texture_ptr, 0, 0, cuda_buffer, size_texture, cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &cuda_texture, 0);
}

void draw() {
	generateCUDAImage();
	glfwPollEvents();

	// clear the color buffer
	glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// bind texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, opengl_texture);

	// we are going to use compiled shaders_program
	glUseProgram(shaders_program);	
	glUniform1i(glGetUniformLocation(shaders_program, "tex"), 0);

	glBindVertexArray(VAO); // binding VAO automatically binds EBO
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0); // unbind VAO

	// swap the screen buffers
	glfwSwapBuffers(window);
}

void initScene() {

	material_t material1(colour_t(0.9f, 0.5f, 0.3f), colour_t(0.7f, 0.1f, 0.9f), 25.0f, 0.0f);
	material_t material2(colour_t(0.4f, 0.6f, 0.8f), colour_t(0.1f, 0.7f, 0.7f), 50.0f, 0.0f);
	material_t material3(colour_t(0.1f, 0.5f, 0.4f), colour_t(1.0f, 0.5f, 0.3f), 50.0f, 0.0f);

	// alloc memory for CPU
	spheres = (sphere_t*)malloc(num_spheres * sizeof(sphere_t));
	lights = (light_t*)malloc(num_lights * sizeof(light_t));
	rays = (ray_t*)malloc(WIDTH*HEIGHT * sizeof(ray_t));
	//buffer = (uint*)malloc(size_texture);

	spheres[0] = sphere_t(material1, point3_t(0.0f, 0.0f, 0.0f), 1.0f);
	spheres[1] = sphere_t(material2, point3_t(1.7f, 0.0f, 0.0f), 0.5f);
	spheres[2] = sphere_t(material2, point3_t(-1.2f, 1.0f, 0.0f), 0.5f);

	lights[0] = light_t(point3_t(1.0f, 1.5f, 3.0f), colour_t(1.0f, 0.0f, 0.0f), 0.7);
	lights[1] = light_t(point3_t(0.0f, 1.0f, 0.0f), colour_t(0.0f, 1.0f, 0.0f), 0.7);
	lights[2] = light_t(point3_t(0.5f, 0.0f, -1.0f), colour_t(0.0f, 0.0f, 1.0f), 0.7);

	// alloc memory for GPU
	//cudaMalloc((void**)&gpu_spheres, num_spheres);
	cudaMalloc((void**)&gpu_rays, WIDTH*HEIGHT);
	cudaMalloc((void**)&gpu_scene_rays, WIDTH*HEIGHT);

	//cudaMemcpy(gpu_spheres, spheres, num_spheres, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(gpu_spheres, spheres, num_spheres*sizeof(sphere_t));
	cudaMemcpyToSymbol(gpu_lights, lights, num_lights*sizeof(light_t));

	generateImageRays<<<grid, block>>>(gpu_rays, WIDTH, HEIGHT, RATIO, FOV);
}

int main(int argc, char** argv)
{
	initGLFW();
	initGL();
	initGLTexture(&opengl_texture, &cuda_texture, WIDTH, HEIGHT);
	initGLShaders("drawtexture.vert", "drawtexture.frag");
	initGLBuffers();
	initCUDABuffers();
	initScene();

	// main loop
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0) {
		draw();
		//break;
		glfwWaitEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	//free(buffer);
	free(rays);
	free(spheres);
	free(lights);
	//free(camera);

	cudaFree(cuda_buffer);
	cudaFree(gpu_rays);
	cudaFree(gpu_scene_rays);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	system("PAUSE");
	return 0;
}