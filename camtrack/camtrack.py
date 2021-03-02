#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import cv2
import click

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    calc_inlier_indices,
)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    MIN_COMMON_IDS = 10
    MAX_ERROR = 5.
    ERROR_TO_REMOVE = 20.
    triangulation_params = TriangulationParameters(
        max_reprojection_error=MAX_ERROR,
        min_triangulation_angle_deg=2.,
        min_depth=0.1
    )

    frame_count = len(corner_storage)
    view_mats = {
        known_view_1[0]: pose_to_view_mat3x4(known_view_1[1]),
        known_view_2[0]: pose_to_view_mat3x4(known_view_2[1]),
    }

    fr, sc = known_view_1[0], known_view_2[0]
    points3d, pids, median_cos = triangulate_correspondences(
        build_correspondences(corner_storage[fr], corner_storage[sc]),
        view_mats[fr], view_mats[sc],
        intrinsic_mat,
        triangulation_params._replace(min_triangulation_angle_deg=0.5)
    )
    pcb = PointCloudBuilder(pids, points3d)

    any_updated = True
    while any_updated:
        any_updated = False

        for i in range(frame_count):
            if i in view_mats:
                continue
            common_ids, ind = snp.intersect(pcb.ids.squeeze(1), corner_storage[i].ids.squeeze(1), indices=True)
            points_cloud = pcb.points[ind[0]]
            points_corners = corner_storage[i].points[ind[1]]

            if len(common_ids) < MIN_COMMON_IDS:
                continue

            click.echo("Processing frame: {}".format(i))

            retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(points_cloud, points_corners.reshape(-1, 1, 2), intrinsic_mat,
                    np.array([]), flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=500, reprojectionError=MAX_ERROR,
                    confidence=0.999)

            if not retval:
                click.echo("Camera position was not found")
                continue
            click.echo("Camera posisition found")
            click.echo("Inliers count: {}".format(len(inliers)))

            view_mats[i] = rodrigues_and_translation_to_view_mat3x4(rvecs, tvecs)
            any_updated = True

            
            outliers_ids = np.delete(common_ids, inliers)
            cnt_added = 0
            for j in view_mats.keys():
                if i == j:
                    continue
                fr, sc = i, j
                points3d, pids, median_cos = triangulate_correspondences(
                    build_correspondences(corner_storage[fr], corner_storage[sc], np.concatenate([pcb.ids, outliers_ids[:, None]])),
                    view_mats[fr], view_mats[sc],
                    intrinsic_mat,
                    triangulation_params
                )
                cnt_added += len(pids)
                pcb.add_points(pids, points3d)

            cnt_removed = 0
            for j in view_mats.keys():
                common_ids, ind = snp.intersect(pcb.ids.squeeze(1), corner_storage[j].ids.squeeze(1), indices=True)
                points_cloud = pcb.points[ind[0]]
                points_corners = corner_storage[j].points[ind[1]]
                inliers = calc_inlier_indices(points_cloud, points_corners, intrinsic_mat @ view_mats[j], ERROR_TO_REMOVE)
                pcb.remove_points(np.delete(common_ids, inliers))
                cnt_removed += len(common_ids) - len(inliers)

            click.echo("PointCloud size: {} Added: {} Removed: {}".format(len(pcb.ids), cnt_added, cnt_removed))

    assert(len(view_mats.keys()) == frame_count and "Not all camera positions found")
        

    view_mats = [view_mats[key] for key in sorted(view_mats.keys())]
    calc_point_cloud_colors(
        pcb,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = pcb.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
