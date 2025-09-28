"""Utility helpers for scenario detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class Segment:
    """Discrete segment on the time axis."""

    start_frame: int
    end_frame: int

    @property
    def length(self) -> int:
        """Return the length of the segment in frames."""

        return self.end_frame - self.start_frame + 1


def find_boolean_segments(
    frames: Sequence[int], mask: Sequence[bool], min_length: int = 1
) -> List[Segment]:
    """Return segments of consecutive frames where mask evaluates to ``True``.

    Parameters
    ----------
    frames:
        Sequence of frame numbers (monotonically increasing).
    mask:
        Boolean sequence aligned with ``frames``.
    min_length:
        Minimum number of consecutive frames required to form a segment.
    """

    if len(frames) != len(mask):
        raise ValueError("frames and mask must have the same length")

    segments: List[Segment] = []
    start_idx: int | None = None
    last_frame: int | None = None

    for idx, (frame, flag) in enumerate(zip(frames, mask)):
        if flag:
            if start_idx is None:
                start_idx = idx
                last_frame = frame
            else:
                if last_frame is not None and frame > last_frame + 1:
                    length = frames[idx - 1] - frames[start_idx] + 1
                    if length >= min_length:
                        segments.append(Segment(frames[start_idx], frames[idx - 1]))
                    start_idx = idx
            last_frame = frame
        else:
            if start_idx is not None:
                length = frames[idx - 1] - frames[start_idx] + 1
                if length >= min_length:
                    segments.append(Segment(frames[start_idx], frames[idx - 1]))
                start_idx = None
                last_frame = None
    if start_idx is not None:
        length = frames[len(frames) - 1] - frames[start_idx] + 1
        if length >= min_length:
            segments.append(Segment(frames[start_idx], frames[len(frames) - 1]))
    return segments
