#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import cv2

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


    triangulation_params = TriangulationParameters(
        max_reprojection_error=5.,
        min_triangulation_angle_deg=0.01,
        min_depth=1.
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
        triangulation_params
    )
    pcb = PointCloudBuilder(pids, points3d)

    print(points3d)
    print(median_cos)

    for i in range(frame_count):
        print(i)
        if i in view_mats:
            print(view_mats[i])
            continue
        
        common_ids, ind = snp.intersect(pcb.ids.squeeze(1), corner_storage[i].ids.squeeze(1), indices=True)
        points_cloud = pcb.points[ind[0]]
        points_corners = corner_storage[i].points[ind[1]]

        normalized_points_corners = cv2.undistortPoints(
            points_corners.reshape(-1, 1, 2),
            intrinsic_mat,
            np.array([])
        ).reshape(-1, 2)

        retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(points_cloud, normalized_points_corners.reshape(-1, 1, 2), intrinsic_mat,
                np.array([]), flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=100, reprojectionError=5., confidence=0.99)
        view_mats[i] = rodrigues_and_translation_to_view_mat3x4(rvecs, tvecs)
        
        print(len(inliers))
        print(view_mats[i])

        fr, sc = i, known_view_1[0]
        points3d, pids, median_cos = triangulate_correspondences(
            build_correspondences(corner_storage[fr], corner_storage[sc], pcb.ids),
            view_mats[fr], view_mats[sc],
            intrinsic_mat,
            triangulation_params
        )
        print(median_cos)
        pcb.add_points(pids, points3d)
        

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
