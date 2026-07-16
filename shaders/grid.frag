#version 460 core

uniform vec2 u_camera_pos;
uniform float u_zoom;
uniform vec2 u_resolution;

in vec2 uv;
out vec4 fragColor;

void main() {
	// World origin pos on screen in pixels
	vec2 origin_screen_px = (u_resolution * 0.5) - (u_camera_pos * u_zoom);

	// How wide grid intervals are in world units
	float approx_major_step = (u_resolution.x / u_zoom) / 5.0;
	if (approx_major_step < 0.0001) {
		approx_major_step = 0.0001;
	}

	float log10_step = floor(log(approx_major_step) / log(10.0));
	float pow10 = pow(10.0, log10_step);
	float ratio = approx_major_step / pow10;

	float multiplier = 1.0;
	if (ratio >= 5.0) {
		multiplier = 5.0;
	} else if (ratio >= 2.0) {
		multiplier = 2.0;
	}

	float major_step = multiplier * pow10;
	float minor_step = major_step / 5.0;

	// Convert grid intervals from world units to screen pixel sizes
	float minor_step_px = minor_step * u_zoom;
	float major_step_px = major_step * u_zoom;

	// Get the current pixel's coord as int
	int current_pixel_x = int(gl_FragCoord.x);
	int current_pixel_y = int(gl_FragCoord.y);

	// Get origin position as plain ints
	int origin_x = int(origin_screen_px.x);
	int origin_y = int(origin_screen_px.y);

	// Track whether this pixel should be a line
	bool is_axis = false;
	bool is_major = false;
	bool is_minor = false;

	// --- AXIS LOGIC (2 px Wide) ---
	// If pixel is exactly on origin column, or 1 pixel to the right
	if (current_pixel_x == origin_x || current_pixel_x == (origin_x + 1)) {
		is_axis = true;
	}
	// If pixel is exactly on the origin row, or 1 pixel above
	if (current_pixel_y == origin_y || current_pixel_y == (origin_y + 1)) {
		is_axis = true;
	}

	// --- MAJOR GRID LOGIC (1 px Wide) ---
	// Calculate distance from origin in pixels
	float dx_from_origin = gl_FragCoord.x - origin_screen_px.x;
	float dy_from_origin = gl_FragCoord.y - origin_screen_px.y;

	// Find closest major grid line index
	int closest_major_line_x = int(round(dx_from_origin / major_step_px));
	int closest_major_line_y = int(round(dy_from_origin / major_step_px));

	// Calculate integer screen pixel where grid line belongs
	int target_major_px_x = origin_x + int(round(float(closest_major_line_x) * major_step_px));
	int target_major_px_y = origin_y + int(round(float(closest_major_line_y) * major_step_px));

	// If current pixel index matches target pixel index, turn it on
	if (current_pixel_x == target_major_px_x || current_pixel_y == target_major_px_y) {
		is_major = true;
	}

	// --- MINOR GRID LOGIC (1 px Wide) ---
	int closest_minor_line_x = int(round(dx_from_origin / minor_step_px));
	int closest_minor_line_y = int(round(dy_from_origin / minor_step_px));

	int target_minor_px_x = origin_x + int(round(float(closest_minor_line_x) * minor_step_px));
	int target_minor_px_y = origin_y + int(round(float(closest_minor_line_y) * minor_step_px));

	if (current_pixel_x == target_minor_px_x || current_pixel_y == target_minor_px_y) {
		is_minor = true;
	}

	// --- DENSITY FADE FOR MINOR LINES ---
	if (minor_step_px < 6.0) {
		is_minor = false; 
	}

	// --- FINAL COLOR RESOLUTION ---
	vec4 final_color = vec4(0.0, 0.0, 0.0, 1.0);    // Default: Black background

	if (is_minor) {
		final_color = vec4(0.12, 0.12, 0.12, 1.0);  // #1F1F1F
	}
	if (is_major) {
		final_color = vec4(0.4, 0.4, 0.4, 1.0);     // #666666
	}
	if (is_axis) {
		final_color = vec4(0.9, 0.9, 0.9, 1.0);     // #E6E6E6
	}

	fragColor = final_color;
}
