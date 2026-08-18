import json
from pathlib import Path

import xmltodict

# '@' symbol marks XML attributes
dummy_data_4xml: dict = {
	"data": {
		"settings": {"column_major": False, "z_order_draw": False, "ticks_per_second": 60},
		"entities": {
			"entity-0": {
				"@type": "points",
				"@color": "#38babf",
				"@dimensions": 2,
				"@point_count": 4,
				"points": " ".join(str(x) for x in [5, 5, -5, 5, -5, -5, 5, -5]),
				"transformations": [
					{"type": "linear", "values": " ".join(x for x in ["cos(1)", "sin(1)", "-sin(1)", "cos(1)"])},
				],
			},
			"entity-1": {
				"@type": "polygon",
				"@color": "#ba38bf",
				"@dimensions": 2,
				"@point_count": 4,
				"points": " ".join(str(x) for x in [5, 5, -5, 5, -5, -5, 5, -5]),
				"transformations": [
					{"type": "linear", "values": " ".join(x for x in ["cos(1)", "-sin(1)", "sin(1)", "cos(1)"])},
				],
			},
		},
	}
}

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


def save_dict_as_xml(data: dict) -> None:
	xml_string = xmltodict.unparse(data, pretty=True)
	save_path = Path(__file__).parent.parent / "save_data.xml"
	with open(save_path, "w") as file:
		file.write(xml_string)


def save_dict_as_json(data: dict) -> None:
	save_path = Path(__file__).parent.parent / "save_data.json"
	with open(save_path, "w", encoding="utf-8") as file:
		json.dump(data, file, separators=(',', ":"))


if __name__ == "__main__":
	save_dict_as_xml(dummy_data_4xml)
	save_dict_as_json(dummy_data_4json)
