from __future__ import annotations

from pathlib import Path
import PIL.ImageDraw
import PIL.Image

import pixellab
from pixellab.models.keypoint import SkeletonLabel
from dotenv import load_dotenv
import os
import json

# Load environment variables and initialize client


def test_estimate_skeleton():
    load_dotenv("../.env")
    image_path = "megaman_zero.png"
    client = pixellab.Client(secret=os.getenv("PIXELLAB_API_KEY"))
    images_dir = Path("tests") / "images"
    test_image = PIL.Image.open(images_dir / image_path)
    permitted_sizes = [16, 32, 64, 128, 256]
    if (
        test_image.size[0] != test_image.size[1]
        and test_image.size[0] not in permitted_sizes
    ):
        test_image = test_image.resize((permitted_sizes[2], permitted_sizes[2]))

    response = client.estimate_skeleton(
        image=test_image,
    )

    assert isinstance(response.keypoints, list)
    assert len(response.keypoints) == 18
    for keypoint in response.keypoints:
        assert isinstance(keypoint["x"], float)
        assert isinstance(keypoint["y"], float)
        assert isinstance(keypoint["label"], str)
        assert isinstance(keypoint["z_index"], float)

    skeleton_dir = Path("tests") / "skeleton_points"
    skeleton_dir.mkdir(exist_ok=True)
    with open(images_dir / f"{image_path.split('.')[0]}_skeleton.json", "w") as f:
        json.dump(response.keypoints, f)


if __name__ == "__main__":
    test_estimate_skeleton()
