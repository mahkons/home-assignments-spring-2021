#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import torch
import numpy as np
import itertools
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
    eye3x4,
    Correspondences,
)


def _calc_loss(points2d, points3d, proj_mat):
    points3d_padded = torch.nn.functional.pad(points3d, (0, 1), 'constant', value=1)
    points_projected = torch.matmul(proj_mat, points3d_padded[:, :, None]).squeeze(2)
    points_projected_norm = points_projected / points_projected[:, 2].unsqueeze(1)
    return torch.nn.functional.smooth_l1_loss(points2d, points_projected_norm[:, :2])
    #  return torch.nn.functional.mse_loss(points2d, points_projected_norm[:, :2])


class Rodrigues(torch.autograd.Function):
    @staticmethod
    def forward(self, inp):
        pose = inp.detach().cpu().numpy()
        rotm, part_jacob = cv2.Rodrigues(pose)
        self.jacob = torch.Tensor(np.transpose(part_jacob)).contiguous()
        rotation_matrix = torch.Tensor(rotm.ravel())
        return rotation_matrix.view(3,3)

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.reshape(1,-1)
        grad_input = torch.mm(grad_output, self.jacob)
        grad_input = grad_input.view(-1)
        return grad_input


def adjust(corner_storage: CornerStorage, intrinsic_mat, pids, points3d, view_mats, steps, update_points=True):
    points3d = torch.tensor(points3d, dtype=torch.float, requires_grad=update_points)
    rt_vecs = {k: torch.tensor(np.concatenate([cv2.Rodrigues(v[:, :3])[0].squeeze(1), v[:, 3]]), dtype=torch.float, requires_grad=True)
        for k, v in view_mats.items()}
    intrinsic_mat = torch.tensor(intrinsic_mat, dtype=torch.float)
    optimizer = torch.optim.Adam(itertools.chain([points3d] if update_points else [], rt_vecs.values()), lr=1e-3)

    for _ in range(steps):
        losses = list()
        for i, rt_vec in rt_vecs.items():
            common_ids, ind = snp.intersect(pids.squeeze(1), corner_storage[i].ids.squeeze(1), indices=True)
            points_cloud = points3d[ind[0]]
            points_corners = torch.tensor(corner_storage[i].points[ind[1]], dtype=torch.float)
            r, t = rt_vec[:3], rt_vec[3:]
            R = Rodrigues.apply(r)
            view_mat = torch.cat((R, t.unsqueeze(1)), dim=1)
            losses.append(_calc_loss(points_corners, points_cloud, intrinsic_mat @ view_mat))

        loss = sum(losses) / len(losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return points3d.detach().numpy(), {k: np.concatenate((cv2.Rodrigues(v[:3].detach().numpy())[0], v[3:].detach().numpy().reshape(3, 1)), axis=1) for k, v in rt_vecs.items()}
    

def choose_best_view(points_1, points_2, ids, intrinsic_mat, poses):
    view_mat_1 = eye3x4()
    triangulation_params = TriangulationParameters(
        max_reprojection_error=3.,
        min_triangulation_angle_deg=2.,
        min_depth=0.01
    )
    mx = -1
    for R, t in poses:
        view_mat_2 = np.hstack((R, t))
        points3d, pids, median_cos = triangulate_correspondences(
            Correspondences(ids, points_1, points_2),
            view_mat_1, view_mat_2,
            intrinsic_mat,
            triangulation_params
        )
        if len(pids) > mx:
            mx = len(pids)
            Rans, tans = R, t

    print(len(ids), mx)
    return Rans, tans, mx


def init_camera_views(corner_storage: CornerStorage, intrinsic_mat: np.ndarray):
    frame_count = len(corner_storage)
    mx = -1
    for i in range(0, frame_count, 3):
        for j in range(i + 5, min(frame_count, i + 50), 3):
            click.echo("Calc initial position: {} {}".format(i, j))
            common_ids, ind = snp.intersect(corner_storage[i].ids.squeeze(1), corner_storage[j].ids.squeeze(1), indices=True)
            if len(common_ids) < 10:
                continue

            points_1 = corner_storage[i].points[ind[0]]
            points_2 = corner_storage[j].points[ind[1]]
            essential, e_inliers = cv2.findEssentialMat(points_1, points_2, intrinsic_mat, method=cv2.RANSAC, threshold=0.1)
            homography, h_inliers = cv2.findHomography(points_1, points_2, method=cv2.RANSAC, ransacReprojThreshold=0.1)
            e_inliers = (e_inliers.squeeze(1) == 1)
            h_inliers = (h_inliers.squeeze(1) == 1)
            if h_inliers.sum() > 0.3 * e_inliers.sum():
                click.echo("homography =(")
                continue

            R1, R2, t = cv2.decomposeEssentialMat(essential)
            
            R, t, val = choose_best_view(points_1[e_inliers], points_2[e_inliers], common_ids[e_inliers], intrinsic_mat,
                    [(R1, t), (R2, t), (R1, -t), (R2, -t)])

            if val > mx:
                mx = val
                bR, bt = R, t
                bi, bj = i, j

            if mx < 50:
                click.echo("not good enough")
                continue

            click.echo("found initial position")
            return (i, view_mat3x4_to_pose(eye3x4())), (j, view_mat3x4_to_pose(np.hstack((R, t))))
    #  assert(False and "Valid initial views not found")
    return (bi, view_mat3x4_to_pose(eye3x4())), (bj, view_mat3x4_to_pose(np.hstack((bR, bt))))


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:

    torch.autograd.set_detect_anomaly(True)
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    #  if known_view_1 is None or known_view_2 is None:
    known_view_1, known_view_2 = init_camera_views(corner_storage, intrinsic_mat)


    MIN_COMMON_IDS = 10
    MAX_ERROR = 3.
    ERROR_TO_REMOVE = 10.
    triangulation_params = TriangulationParameters(
        max_reprojection_error=MAX_ERROR,
        min_triangulation_angle_deg=2.,
        min_depth=0.01
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
        triangulation_params._replace(min_triangulation_angle_deg=2.)
    )
    pcb = PointCloudBuilder(pids, points3d)
    #  points3d, view_mats = adjust(corner_storage, intrinsic_mat, pcb.ids, pcb.points, view_mats, 200)
    #  pcb.update_points(pcb.ids, points3d)

    any_updated = True
    STEPS = 500
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

            #  points3d, n_view_mats = adjust(corner_storage, intrinsic_mat, pcb.ids, pcb.points, {i: view_mats[i]}, STEPS, update_points=False)
            #  view_mats[i] = n_view_mats[i]
            #  pcb.update_points(pcb.ids, points3d)

            outliers_ids = np.delete(common_ids, inliers)
            cnt_added = 0
            for j in view_mats.keys():
                if i == j:
                    continue
                fr, sc = i, j
                correspondences = build_correspondences(corner_storage[fr], corner_storage[sc], np.concatenate([pcb.ids, outliers_ids[:, None]]))
                if len(correspondences.ids) == 0:
                    continue
                points3d, pids, median_cos = triangulate_correspondences(
                    correspondences,
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
            #  if (i + 1) % max(1, frame_count // 5) == 0:
                #  points3d, view_mats = adjust(corner_storage, intrinsic_mat, pcb.ids, pcb.points, view_mats, STEPS)
                #  pcb.update_points(pcb.ids, points3d)

    #  assert(len(view_mats.keys()) == frame_count and "Not all camera positions found")
    for i in range(frame_count):
        for k in range(frame_count):
            if i + k in view_mats:
                view_mats[i] = view_mats[i + k]
                break
            if i - k in view_mats:
                view_mats[i] = view_mats[i - k]
                break

    points3d, view_mats = adjust(corner_storage, intrinsic_mat, pcb.ids, pcb.points, view_mats, STEPS)
    pcb.update_points(pids, points3d)
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
