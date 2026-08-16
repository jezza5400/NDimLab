#version 460 core

layout (std430, binding = 0) readonly buffer PointBuffer {
	float rawCoords[];
};

layout (std430, binding = 1) readonly buffer TransformBuffer {
	float transMatrix[];
};

uniform uint u_dimension;
uniform vec4 u_pointColor;
uniform vec2 u_camera_pos;
uniform float u_zoom;
uniform vec2 u_resolution;

out vec4 vertColor;

void main() {
	uint startIndex = gl_VertexID * u_dimension;

	vec4 clipPos = vec4(0.0);

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

	uint wRow = u_dimension * 4u;
	clipPos += vec4(
		transMatrix[wRow + 0u],
		transMatrix[wRow + 1u],
		transMatrix[wRow + 2u],
		transMatrix[wRow + 3u]
	);

	vec2 ndc_scale = (2.0 * u_zoom) / u_resolution;
	clipPos.xy = ndc_scale * (clipPos.xy - u_camera_pos);

	const float PI_OVER_2 = acos(0.0);
	const float Z_COMPRESS = 4.;  // larger = flatter falloff = more usable Z range
	clipPos.z = -atan(clipPos.z / Z_COMPRESS) / PI_OVER_2;  // Compress z into [-1, 1] keeping +z pointing out screen

	gl_Position = clipPos;
	vertColor = u_pointColor;
	gl_PointSize = 8.0;
}
