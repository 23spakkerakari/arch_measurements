"""LabelMe -> ground_truth conversion."""

import json

import pytest

from arqen_validation.labelme import convert_labelme_document

pytestmark = pytest.mark.unit


def test_convert_room_wall_door():
    doc = {
        "imageWidth": 1000,
        "imageHeight": 800,
        "shapes": [
            {
                "label": "Room",
                "shape_type": "rectangle",
                "points": [[100, 100], [400, 300]],
            },
            {
                "label": "Wall",
                "shape_type": "rectangle",
                "points": [[100, 95], [400, 105]],
            },
            {
                "label": "Door",
                "shape_type": "rectangle",
                "points": [[200, 100], [240, 130]],
            },
            {
                "label": "Toilet",
                "shape_type": "rectangle",
                "points": [[10, 10], [20, 20]],
            },
        ],
    }
    gt, report = convert_labelme_document(doc, "test_case")
    assert len(gt["rooms"]) == 1
    assert gt["rooms"][0]["bbox_px"] == [100, 100, 400, 300]
    assert len(gt["walls"]) == 1
    w = gt["walls"][0]["px_coords"]
    assert abs(w[1] - w[3]) < 12.0  # roughly horizontal centerline band
    assert abs((w[1] + w[3]) / 2 - 100) < 6.0
    assert len(gt["doors"]) == 1
    assert report["skipped_labels"]["Toilet"] == 1


def test_real_labelme_stub():
    path = (
        "C:/Users/jakep/Downloads/arqen-labs-jun/arqen-labs-jun/"
        "floor-plan-annotated/FP_86_2.json"
    )
    try:
        doc = json.loads(open(path, encoding="utf-8").read())
    except FileNotFoundError:
        pytest.skip("LabelMe sample not on this machine")
    gt, _ = convert_labelme_document(doc, "fp_86_2")
    assert len(gt["rooms"]) == 2
    assert len(gt["walls"]) == 2
    assert gt["image_size_px"] == [1422, 742]
