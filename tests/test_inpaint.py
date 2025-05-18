from __future__ import annotations

from pathlib import Path

import PIL.Image

import pixellab
from dotenv import load_dotenv
import os


def test_inpaint():
    load_dotenv("../.env")

    client = pixellab.Client(secret=os.getenv("PIXELLAB_API_KEY"))
    resize_dim = (128, 128)
    image_name = "boy.png"
    mask_name = "mask.png"
    images_dir = Path("tests") / "images"
    inpainting_image = PIL.Image.open(images_dir / image_name).resize(resize_dim)
    mask_image = PIL.Image.open(images_dir / mask_name).resize(resize_dim)

    response = client.inpaint(
        description="boy with wings",
        image_size=dict(width=resize_dim[0], height=resize_dim[1]),
        no_background=True,
        inpainting_image=inpainting_image,
        mask_image=mask_image,
        text_guidance_scale=3.0,
    )

    image = response.image.pil_image()
    assert isinstance(image, PIL.Image.Image)
    assert image.size == resize_dim

    results_dir = Path("tests") / "results"
    results_dir.mkdir(exist_ok=True)

    image.save(
        results_dir
        / f"inpainting_{image_name.split('.')[0]}_with_{mask_name.split('.')[0]}.png"
    )


if __name__ == "__main__":
    test_inpaint()
