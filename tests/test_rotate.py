from __future__ import annotations

from pathlib import Path

import PIL.Image

import pixellab
from dotenv import load_dotenv
import os


def test_rotate():
    load_dotenv("../.env")
    client = pixellab.Client(secret=os.getenv("PIXELLAB_API_KEY"))
    resize_dim = (128, 128)
    image_name = "fire.jpeg"
    from_direction = "south"
    to_direction = "east"
    from_view = "side"
    to_view = "side"
    images_dir = Path("tests") / "images"
    reference_image = PIL.Image.open(images_dir / image_name).resize(resize_dim)
    init_image = reference_image

    response = client.rotate(
        from_direction=from_direction,
        from_view=from_view,
        to_direction=to_direction,
        to_view=to_view,
        image_size={"width": resize_dim[0], "height": resize_dim[1]},
        image_guidance_scale=7.5,
        from_image=reference_image,
        init_image=init_image,
        init_image_strength=10,
    )

    image = response.image.pil_image()
    assert isinstance(image, PIL.Image.Image)
    assert image.size == resize_dim

    results_dir = Path("tests") / "results"
    results_dir.mkdir(exist_ok=True)

    image.save(
        results_dir
        / f"rotation_{image_name.split('.')[0]}_{from_direction}_to_{to_direction}.png"
    )


if __name__ == "__main__":
    test_rotate()
