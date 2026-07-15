#version 460 core

uniform vec2 u_camera_pos;
uniform float u_zoom;
uniform vec2 u_resolution;

in vec2 uv;

out vec4 fragColor;

float line_mask(float dist_px, float half_width_px) {
	return step(dist_px, half_width_px);
}

void main() {
	// Reconstruct world coordinates
	vec2 world_coord = (uv * (u_resolution / 2.0) / u_zoom) + u_camera_pos;

	// Calculate approximately how wide a 5th of screen is in world units
	float approx_major_step = (u_resolution.x / u_zoom) / 5.0;

	// prevent division-by-zero
	approx_major_step = max(approx_major_step, 0.0001);

	// Find nearest power of 10
	float log10_step = floor(log(approx_major_step) / log(10.0));
	float pow10 = pow(10.0, log10_step);
	float ratio = approx_major_step / pow10;

	float low_to_mid = step(2.0, ratio);  // 1.0 if ratio >= 2.0, else 0.0
	float mid_to_high = step(5.0, ratio); // 1.0 if ratio >= 5.0, else 0.0

	// Linearly interpolate multipliers based on step tests
	float multiplier = mix(1.0, 2.0, low_to_mid);
	multiplier = mix(multiplier, 5.0, mid_to_high);

	float major_step = multiplier * pow10;
	float minor_step = major_step / 5.0;

	// Distances to lines in screen pixels so cosmetic widths stay constant.
	vec2 dist_to_minor_px = abs(fract(world_coord / minor_step - 0.5) - 0.5) * minor_step * u_zoom;
	vec2 dist_to_major_px = abs(fract(world_coord / major_step - 0.5) - 0.5) * major_step * u_zoom;
	vec2 axis_center_px = (u_resolution * 0.5) - (u_camera_pos * u_zoom);
	vec2 dist_to_axis_px = abs(gl_FragCoord.xy - axis_center_px);

	// Cosmetic line widths, expressed in pixels.
	const float minor_half_width_px = 0.5;
	const float major_half_width_px = 0.5;
	const float axis_half_width_px = 1.0;

	// Line masks.
	float is_minor = max(line_mask(dist_to_minor_px.x, minor_half_width_px), line_mask(dist_to_minor_px.y, minor_half_width_px));
	float is_major = max(line_mask(dist_to_major_px.x, major_half_width_px), line_mask(dist_to_major_px.y, major_half_width_px));
	float is_axis = max(line_mask(dist_to_axis_px.x, axis_half_width_px), line_mask(dist_to_axis_px.y, axis_half_width_px));

	// Define colors
	vec4 color_bg = vec4(0.0, 0.0, 0.0, 1.0);       // Black background
	vec4 color_minor = vec4(0.12, 0.12, 0.12, 1.0); // #1F1F1F
	vec4 color_major = vec4(0.4, 0.4, 0.4, 1.0);    // #666666
	vec4 color_axis = vec4(0.9, 0.9, 0.9, 1.0);     // #E6E6E6

	// Layer composition
	vec4 final_color = mix(color_bg, color_minor, is_minor);
	final_color = mix(final_color, color_major, is_major);
	final_color = mix(final_color, color_axis, is_axis);

	fragColor = final_color;
}
