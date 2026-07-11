#version 460 core

uniform mat4 u_trans;
uniform float u_aspect;

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inColor;

out vec3 vertColor;

void main() {
	vec4 rotatedPoint = u_trans * vec4(inPosition, 1.0);
	rotatedPoint.x /= u_aspect;
	gl_Position = rotatedPoint;
	vertColor = inColor;
}