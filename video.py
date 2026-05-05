"""Create a video from saved displacement plots."""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image


def make_video(
    plot_dir: str | Path = "plots",
    output_path: str | Path = "displacement_evolution.mp4",
    fps: int = 10,
):
    """Create an MP4 video from saved plot images."""
    plot_dir = Path(plot_dir)
    output_path = Path(output_path)

    frames = sorted(plot_dir.glob("w_epoch_*.png"))
    if not frames:
        print("No plot frames found.")
        return

    # Read first frame to get dimensions
    first = Image.open(frames[0])
    fig, ax = plt.subplots(figsize=(first.width / 150, first.height / 150), dpi=150)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    im = ax.imshow(first)

    def update(frame_path):
        im.set_data(Image.open(frame_path))
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=1000 // fps, blit=True
    )
    ani.save(str(output_path), writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"Video saved to {output_path} ({len(frames)} frames)")


if __name__ == "__main__":
    make_video()
