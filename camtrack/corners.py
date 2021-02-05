#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

import itertools

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    MAX_CORNERS = 5000
    BLOCK_SIZE = 7
    MIN_DISTANCE = 11
    QUALITY_LEVEL = 0.01

    image_0 = frame_sequence[0]
    corners = cv2.goodFeaturesToTrack(image_0, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL,
            minDistance=MIN_DISTANCE, blockSize=BLOCK_SIZE, useHarrisDetector=False).squeeze(1)

    ids_generator = itertools.count()
    ids = np.array([next(ids_generator) for _ in range(len(corners))])
    frame_corners = FrameCorners(ids, corners, np.ones_like(ids) * BLOCK_SIZE)

    builder.set_corners_at_frame(0, frame_corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        corners, status, error = cv2.calcOpticalFlowPyrLK((image_0 * 255).astype(np.uint8),
                (image_1 * 255).astype(np.uint8), corners, None, winSize=(15, 15), maxLevel=3, minEigThreshold=0.001)
        status = status.squeeze(1)

        corners = corners[status == 1]
        ids = ids[status == 1]

        new_corners = cv2.goodFeaturesToTrack(image_1, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL,
                minDistance=MIN_DISTANCE, corners=None, mask=None, blockSize=BLOCK_SIZE, useHarrisDetector=False)
        if new_corners is None:
            new_corners = np.empty((0, 2), corners.dtype)
        else:
            new_corners = new_corners.squeeze(1)

        dist = np.linalg.norm(corners[None, :] - new_corners[:, None], axis=2)
        new_corners = new_corners[np.min(dist, axis=1) >= MIN_DISTANCE, :]

        corners = np.concatenate((corners, new_corners))
        ids = np.concatenate((ids, np.array([next(ids_generator) for _ in range(len(new_corners))], dtype=np.int32)))

        frame_corners = FrameCorners(ids, corners, np.ones_like(ids) * BLOCK_SIZE)
        builder.set_corners_at_frame(frame, frame_corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
