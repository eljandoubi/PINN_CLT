"""Create a video from saved displacement plots.

This uses imageio to write an MP4. If imageio/ffmpeg is unavailable it will
fall back to saving a GIF using Pillow.
"""

from pathlib import Path
from typing import Union

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from PIL.Image import Resampling


def _normalize_frame(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize frame to target (H, W) using Pillow for consistent dimensions."""
    pil_img = Image.fromarray(img).resize(
        (target_size[1], target_size[0]), Resampling.LANCZOS
    )
    return np.array(pil_img)


def make_video(
    plot_dir: Union[str, Path] = "plots",
    output_path: Union[str, Path] = "displacement_evolution.mp4",
    fps: int = 10,
):
    plot_dir = Path(plot_dir)
    output_path = Path(output_path)

    # Prefer 3D frames if present (named w_epoch_XXXXXX_3d.png)
    frames = sorted(plot_dir.glob("w_epoch_*_3d.png"))
    if not frames:
        # Fall back to any w_epoch_*.png
        frames = sorted(plot_dir.glob("w_epoch_*.png"))
    if not frames:
        print("No plot frames found.")
        return

    try:
        # Read first frame to determine target size (divisible by 16)
        first = imageio.imread(frames[0])
        h, w = first.shape[:2]
        new_h = int(np.ceil(h / 16) * 16)
        new_w = int(np.ceil(w / 16) * 16)
        target_size = (new_h, new_w)

        with imageio.get_writer(
            str(output_path), fps=fps, codec="libx264", macro_block_size=1
        ) as writer:
            for f in frames:
                img = imageio.imread(f)
                img = _normalize_frame(img, target_size)
                writer.append_data(img)  # type: ignore[attr-defined]
        print(f"Video saved to {output_path} ({len(frames)} frames)")
    except Exception as e:
        print(f"MP4 writer failed ({e}), falling back to GIF output.")
        # Fallback: create animated GIF using Pillow
        imgs = [Image.open(f).convert("RGBA") for f in frames]
        gif_path = output_path.with_suffix(".gif")
        imgs[0].save(
            gif_path,
            save_all=True,
            append_images=imgs[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"GIF saved to {gif_path} ({len(frames)} frames)")


if __name__ == "__main__":
    make_video()
