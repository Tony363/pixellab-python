from __future__ import annotations

from pathlib import Path

import PIL.Image

import pixellab
from dotenv import load_dotenv
import os


def test_rotate():
    load_dotenv("../.env")
    client = pixellab.Client(secret=os.getenv("PIXELLAB_API_KEY"))
    images_dir = Path("tests") / "images"
    reference_image = PIL.Image.open(images_dir / "boy.png").resize((16, 16))
    init_image = reference_image

    response = client.rotate(
        from_direction="south",
        from_view="side",
        to_direction="north",
        to_view="side",
        image_size={"width": 16, "height": 16},
        image_guidance_scale=7.5,
        from_image=reference_image,
        init_image=init_image,
        init_image_strength=10,
    )

    image = response.image.pil_image()
    assert isinstance(image, PIL.Image.Image)
    assert image.size == (16, 16)

    results_dir = Path("tests") / "results"
    results_dir.mkdir(exist_ok=True)

    image.save(results_dir / "rotation_boy_south_to_north.png")


if __name__ == "__main__":
    test_rotate()
