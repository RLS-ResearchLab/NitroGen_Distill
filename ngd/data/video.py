"""Frame extraction for dataset chunks.

The HF dataset ships only action labels; the source videos must be fetched separately
(see scripts/download_dataset.sh for labels, yt-dlp for videos). Given a local video
file and a chunk's metadata.json, this module extracts the frames that align 1:1 with
the parquet action rows, applies the game-area crop, and resizes for the vision tower.
"""

from fractions import Fraction
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


def extract_chunk_frames(
    video_path: str | Path,
    metadata: dict,
    frame_indices: Optional[list[int]] = None,  # indices *within the chunk*; None = all
    size: Optional[tuple[int, int]] = (256, 256),  # (H, W) output; None = no resize
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (chunk_frame_index, RGB uint8 HxWx3) aligned with parquet rows.

    Alignment: parquet row ``i`` of a chunk corresponds to original-video frame
    ``metadata.original_video.start_frame + i``.
    """
    import av

    original = metadata["original_video"]
    start_frame = original["start_frame"]
    n_frames = metadata["chunk_size"]
    wanted = set(range(n_frames)) if frame_indices is None else set(frame_indices)

    crop = _game_area_pixels(metadata)

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = Fraction(stream.average_rate)

        # Seek near the chunk start, then decode forward.
        start_pts = int(start_frame / fps / stream.time_base)
        container.seek(start_pts, stream=stream, backward=True)

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            abs_idx = round(float(frame.pts * stream.time_base) * float(fps))
            chunk_idx = abs_idx - start_frame
            if chunk_idx < 0 or chunk_idx not in wanted:
                if chunk_idx >= n_frames:
                    break
                continue

            img = frame.to_ndarray(format="rgb24")
            if crop is not None:
                x0, y0, x1, y1 = crop
                img = img[y0:y1, x0:x1]
            if size is not None:
                img = _resize(img, size)
            yield chunk_idx, img

            wanted.discard(chunk_idx)
            if not wanted:
                break


def _game_area_pixels(metadata: dict) -> Optional[tuple[int, int, int, int]]:
    """bbox_game_area (relative [0,1] coords) -> (x0, y0, x1, y1) pixels, or None."""
    bbox = metadata.get("bbox_game_area")
    if not bbox:
        return None
    h, w = metadata["original_video"]["resolution"]
    return (
        int(bbox["xtl"] * w),
        int(bbox["ytl"] * h),
        int(bbox["xbr"] * w),
        int(bbox["ybr"] * h),
    )


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((size[1], size[0]), Image.BILINEAR))
