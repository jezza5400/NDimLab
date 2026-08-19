import json
from pathlib import Path

dummy_data_4json: dict = {
	"settings": {"column_major": False, "z_order_draw": False, "ticks_per_second": 60},
	"entities": [
		{
			"id": "entity-0",
			"type": "points",
			"color": "#38babf",
			"dimensions": 2,
			"point_count": 4,
			"points": [5, 5, -5, 5, -5, -5, 5, -5],
			"transformations": [{"type": "linear", "values": ["cos(1)", "sin(1)", "-sin(1)", "cos(1)"]}],
		}
	],
}


def save_dict_as_json(data: dict, minify: bool = True) -> None:
	save_path = Path(__file__).parent.parent / "save_data.json"
	with open(save_path, "w", encoding="utf-8") as file:
		if minify:
			json.dump(data, file, separators=(",", ":"))
		else:
			json.dump(data, file, indent="\t")


if __name__ == "__main__":
	save_dict_as_json(dummy_data_4json, minify=False)
