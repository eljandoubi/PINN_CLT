"""Tests for video.py — frame-to-video assembly with GIF fallback."""

from pathlib import Path

import numpy as np
from PIL import Image

import video
from video import _normalize_frame, make_video


def _make_frames(directory: Path, count: int, suffix: str = "_3d") -> None:
    for i in range(1, count + 1):
        img = Image.new("RGB", (64, 64), color=(i * 20 % 256, 100, 150))
        img.save(directory / f"w_epoch_{i:06d}{suffix}.png")


class TestNormalizeFrame:
    def test_resizes_to_target_shape(self):
        img = np.zeros((30, 20, 3), dtype=np.uint8)
        out = _normalize_frame(img, target_size=(48, 32))
        assert out.shape[:2] == (48, 32)

    def test_preserves_channels(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        out = _normalize_frame(img, target_size=(16, 16))
        assert out.shape == (16, 16, 3)


class TestMakeVideo:
    def test_no_frames_creates_no_output(self, tmp_path, capsys):
        output = tmp_path / "out.mp4"
        make_video(plot_dir=tmp_path, output_path=output)
        assert not output.exists()
        assert "No plot frames found" in capsys.readouterr().out

    def test_creates_mp4_from_3d_frames(self, tmp_path):
        _make_frames(tmp_path, count=3, suffix="_3d")
        output = tmp_path / "video.mp4"
        make_video(plot_dir=tmp_path, output_path=output, fps=5)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_falls_back_to_plain_frames_when_no_3d(self, tmp_path):
        _make_frames(tmp_path, count=2, suffix="")
        output = tmp_path / "video.mp4"
        make_video(plot_dir=tmp_path, output_path=output, fps=5)
        assert output.exists()

    def test_prefers_3d_frames_over_plain(self, tmp_path, capsys):
        _make_frames(tmp_path, count=2, suffix="_3d")
        _make_frames(tmp_path, count=5, suffix="")
        make_video(plot_dir=tmp_path, output_path=tmp_path / "video.mp4", fps=5)
        assert "(2 frames)" in capsys.readouterr().out

    def test_falls_back_to_gif_on_writer_failure(self, tmp_path, monkeypatch):
        _make_frames(tmp_path, count=2, suffix="_3d")

        def _raise(*args, **kwargs):
            raise RuntimeError("forced failure for test")

        monkeypatch.setattr(video.imageio, "get_writer", _raise)

        output = tmp_path / "video.mp4"
        make_video(plot_dir=tmp_path, output_path=output, fps=5)

        assert not output.exists()
        assert output.with_suffix(".gif").exists()
