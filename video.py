"""Create a video from saved displacement plots.

This uses imageio to write an MP4. If imageio/ffmpeg is unavailable it will
fall back to saving a GIF using Pillow.
"""

from pathlib import Path
from typing import Union

import imageio
from PIL import Image


def make_video(
    plot_dir: Union[str, Path] = "plots",
    output_path: Union[str, Path] = "displacement_evolution.mp4",
    fps: int = 10,
):
    plot_dir = Path(plot_dir)
    output_path = Path(output_path)

    frames = sorted(plot_dir.glob("w_epoch_*.png"))
    if not frames:
        print("No plot frames found.")
        return

    try:
        # Use imageio writer (uses ffmpeg under the hood if available)
        with imageio.get_writer(str(output_path), fps=fps, codec="libx264") as writer:
            for f in frames:
                img = imageio.imread(f)
                writer.append_data(img)
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
