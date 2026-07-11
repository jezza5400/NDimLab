#version 460 core

layout (location = 0) in vec2 inPosition;

out vec2 uv;

void main() {
	uv = inPosition;
	gl_Position = vec4(inPosition, 0.0, 1.0);
}