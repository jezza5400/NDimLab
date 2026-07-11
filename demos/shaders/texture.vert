#version 460 core

layout (location = 0) in vec2 inPosition;

out vec2 v_texcoord;

void main() {
	gl_Position = vec4(inPosition, 0.0, 1.0);
	v_texcoord = inPosition * 0.5 + 0.5; // Convert NDC [-1, 1] to Texture Coordinates [0, 1]
}
