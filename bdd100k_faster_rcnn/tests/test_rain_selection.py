import csv

from select_night_count_control import select
from select_rain_subsets import rain_class_gaps
from src.config import CATEGORIES


def test_weather_quota_selects_exact_number_of_rainy_images():
    rows = []
    for i in range(12):
        row = {"image_name": f"{i}.jpg", "weather": "rainy" if i < 5 else "clear",
               "scene": "city street", "timeofday": "night" if i % 2 else "daytime",
               "small_box_count": 0, "medium_box_count": 1, "large_box_count": 0}
        row.update({f"count_{c.replace(' ', '_')}": int(c == "car") for c in CATEGORIES})
        rows.append(row)
    chosen = select(rows, budget=8, night_quota=3, seed=42,
                    target_field="weather", target_value="rainy")
    assert len(chosen) == len(set(chosen)) == 8
    assert sum(rows[i]["weather"] == "rainy" for i in chosen) == 3


def test_rain_gap_ignores_classes_absent_from_either_domain(tmp_path):
    path = tmp_path / "metrics.csv"
    fields = ["domain", *[f"ap_{c.replace(' ', '_')}" for c in CATEGORIES]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"domain": "rainy", "ap_car": 0.2, "ap_train": ""})
        writer.writerow({"domain": "nonrain", "ap_car": 0.3, "ap_train": 0.1})
    gaps = rain_class_gaps([path])
    assert abs(gaps["car"] - 0.1) < 1e-10
    assert gaps["train"] == 0.0
