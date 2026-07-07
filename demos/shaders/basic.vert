#version 460 core

uniform mat4 trans;
uniform float aspect;

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inColour;
out vec3 vertColour;

void main() {
	vec4 rotatedPoint = trans * vec4(inPosition, 1.0);
	rotatedPoint.x /= aspect;
	gl_Position = rotatedPoint;
	vertColour = inColour;
}