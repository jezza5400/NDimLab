#version 460 core

layout (std430, binding = 0) readonly buffer PointBuffer {
	float rawCoords[];
};

// (u_dimension + 1) rows x 4 cols, row-major: transMatrix[row * 4 + col]
// Row u_dimension is the translation/affine row (the old homogeneous "1").
layout (std430, binding = 1) readonly buffer TransformBuffer {
	float transMatrix[];
};

uniform uint u_dimension;
uniform vec4 u_pointColor;

out vec4 vertColor;

void main() {
	uint startIndex = gl_VertexID * u_dimension;

	vec4 clipPos = vec4(0.0);

	// Accumulate each input dimension's contribution to clip space.
	// Loop bound is uniform (same for every invocation) -> coherent, not divergent.
	for (uint d = 0u; d < u_dimension; ++d) {
		float coord = rawCoords[startIndex + d];
		uint row = d * 4u;
		clipPos += coord * vec4(
			transMatrix[row + 0u],
			transMatrix[row + 1u],
			transMatrix[row + 2u],
			transMatrix[row + 3u]
		);
	}

	// Affine/translation row (equivalent to the old implicit "w=1" component)
	uint wRow = u_dimension * 4u;
	clipPos += vec4(
		transMatrix[wRow + 0u],
		transMatrix[wRow + 1u],
		transMatrix[wRow + 2u],
		transMatrix[wRow + 3u]
	);

	gl_Position = clipPos;
	vertColor = u_pointColor;
	gl_PointSize = 8.0;
}
