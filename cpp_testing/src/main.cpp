#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GLAD_GL_IMPLEMENTATION
#include <glad/gl.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <vector>

std::string load_shader(const std::filesystem::path &filepath) {
	std::println("Loading shader at: {}", filepath.string());
	std::ifstream shader_file(filepath);
	if (!shader_file.is_open()) {
		std::println(stderr, "Error: Failed to open shader at: {}", filepath.string());
		return "";
	}
	std::stringstream shader_stream;
	shader_stream << shader_file.rdbuf();
	return shader_stream.str();
}

void error_callback([[maybe_unused]] int error, const char *description) {
	std::println(stderr, "Error: {}", description);
}

void close_callback([[maybe_unused]] GLFWwindow *window) {
	std::println("Window close requested");
}

static void key_callback(GLFWwindow *window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}

void framebuffer_size_callback([[maybe_unused]] GLFWwindow *window, int width, int height) {
	glViewport(0, 0, width, height);
}

struct vec2 {
	float x = 0.0f;
	float y = 0.0f;
};

static constexpr std::array<vec2, 6> vertices = {{
	{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f},
	{-1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}
}};

int main() {
	glfwSetErrorCallback(error_callback);

	if (!glfwInit()) {
		std::println(stderr, "glfwInit failed");
		return EXIT_FAILURE;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow *window = glfwCreateWindow(640, 480, "Grid Renderer", NULL, NULL);
	if (!window) {
		std::println(stderr, "glfwCreateWindow failed");
		glfwTerminate();
		return EXIT_FAILURE;
	}

	glfwSetWindowCloseCallback(window, close_callback);
	glfwSetKeyCallback(window, key_callback);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	glfwMakeContextCurrent(window);
	if (!gladLoadGL(glfwGetProcAddress)) {
		std::println(stderr, "Failed to initialize GLAD");
		glfwDestroyWindow(window);
		glfwTerminate();
		return EXIT_FAILURE;
	}
	glfwSwapInterval(0);

	GLuint vertex_array, vertex_buffer;
	glGenVertexArrays(1, &vertex_array);
	glGenBuffers(1, &vertex_buffer);

	glBindVertexArray(vertex_array);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices.data(), GL_STATIC_DRAW);

	std::string grid_vert = load_shader(std::filesystem::path(SHADERS_DIR) / "grid.vert");
	std::string grid_frag = load_shader(std::filesystem::path(SHADERS_DIR) / "grid.frag");
	const char *vert_ptr = grid_vert.c_str();
	const char *frag_ptr = grid_frag.c_str();

	const GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, &vert_ptr, NULL);
	glCompileShader(vertex_shader);

	const GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader, 1, &frag_ptr, NULL);
	glCompileShader(fragment_shader);

	const GLuint program = glCreateProgram();
	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);
	glLinkProgram(program);

	const GLint inPosition_location = glGetAttribLocation(program, "inPosition");
	glEnableVertexAttribArray(inPosition_location);
	glVertexAttribPointer(inPosition_location, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), (void *)0);

	const GLint res_loc = glGetUniformLocation(program, "u_resolution");
	const GLint cam_loc = glGetUniformLocation(program, "u_camera_pos");
	const GLint zoom_loc = glGetUniformLocation(program, "u_zoom");

	while (!glfwWindowShouldClose(window)) {
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		glViewport(0, 0, width, height);

		glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(program);
		glUniform2f(res_loc, static_cast<float>(width), static_cast<float>(height));
		glUniform2f(cam_loc, 0.0f, 0.0f);
		glUniform1f(zoom_loc, 50.0f);

		glBindVertexArray(vertex_array);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glDeleteProgram(program);
	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);
	glDeleteBuffers(1, &vertex_buffer);
	glDeleteVertexArrays(1, &vertex_array);

	glfwDestroyWindow(window);
	glfwTerminate();
	return EXIT_SUCCESS;
}
