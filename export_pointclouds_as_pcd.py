# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Licensed under the Creative Commons [see licence.txt]

"""
Export fused point clouds of a scene to a Wavefront OBJ file.
This point-cloud can be viewed in your favorite 3D rendering tool, e.g. Meshlab or Maya.
"""
import os
import os.path as osp
import argparse
from typing import Tuple

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

from utils.data_classes import LidarPointCloud
from utils.geometry_utils import view_points

from open3d import write_point_cloud, PointCloud, Matrix4dVector, Vector3dVector, Vector3iVector

import sys
# print(sys.path)

from nuscenes.nuscenes import NuScenes


def export_scene_pointcloud(nusc: NuScenes, out_path: str, scene_token: str, scene_name, trafo_in_origin_BOOL: bool, write_ascii_BOOL: bool, channel: str='LIDAR_TOP',
                            min_dist: float=3.0, max_dist: float=30.0, verbose: bool=True) -> None:
    """
    Export fused point clouds of a scene to a Wavefront OBJ file.
    This point-cloud can be viewed in your favorite 3D rendering tool, e.g. Meshlab or Maya.
    :param nusc: NuScenes instance.
    :param out_path: Output path to write the point-cloud to.
    :param scene_token: Unique identifier of scene to render.
    :param channel: Channel to render.
    :param min_dist: Minimum distance to ego vehicle below which points are dropped.
    :param max_dist: Maximum distance to ego vehicle above which points are dropped.
    :param verbose: Whether to print messages to stdout.
    """

    # Check inputs.
    valid_channels = ['LIDAR_TOP', 'RADAR_FRONT', 'RADAR_FRONT_RIGHT', 'RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT',
                      'RADAR_BACK_RIGHT']
    camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    assert channel in valid_channels, 'Input channel {} not valid.'.format(channel)

    # Get records from DB.
    scene_rec = nusc.get('scene', scene_token)
    start_sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
    sd_rec = nusc.get('sample_data', start_sample_rec['data'][channel])

    # Make list of frames
    cur_sd_rec = sd_rec
    sd_tokens = []
    while cur_sd_rec['next'] != '':

        cur_sd_rec = nusc.get('sample_data', cur_sd_rec['next'])
        sd_tokens.append(cur_sd_rec['token'])


    ego_pose_info = []
    zeitstample = 0
    for sd_token in tqdm(sd_tokens):
        zeitstample = zeitstample + 1
        if verbose:
            print('Processing {}'.format(sd_rec['filename']))
        sc_rec = nusc.get('sample_data', sd_token)
        sample_rec = nusc.get('sample', sc_rec['sample_token'])
        # lidar_token = sd_rec['token'] # only for the start_sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        lidar_rec = nusc.get('sample_data', sd_token)
        pc = LidarPointCloud.from_file(osp.join(nusc.dataroot, lidar_rec['filename']))


        # Points live in their own reference frame. So they need to be transformed (DELETED: via global to the image plane.
        # First step: transform the point cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', lidar_rec['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Optional Filter by distance to remove the ego vehicle.
        dists_origin = np.sqrt(np.sum(pc.points[:3, :] ** 2, axis=0))

        keep = np.logical_and(min_dist <= dists_origin, dists_origin <= max_dist)
        pc.points = pc.points[:, keep]
        # coloring = coloring[:, keep]
        if verbose:
            print('Distance filter: Keeping %d of %d points...' % (keep.sum(), len(keep)))



        # Second step: transform to the global frame.
        poserecord = nusc.get('ego_pose', lidar_rec['ego_pose_token'])
        if trafo_in_origin_BOOL == True:
            pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
            pc.translate(np.array(poserecord['translation']))

        ego_pose_info.append("Ego-Pose-Info for time-stample %i sd_token(%s) :\n" % (zeitstample, sd_token))
        ego_pose_info.append("   Rotation : %s \n" % np.array2string(np.array(poserecord['rotation'])))
        ego_pose_info.append("   Translation : %s \n" % np.array2string(np.array(poserecord['translation'])))


        pc_nparray = np.asarray(pc.points)
        # print(pc_nparray.T.shape(1))
        xyzi = pc_nparray.T

        xyz = np.zeros((xyzi.shape[0], 3))
        xyz[:, 0] = np.reshape(pc_nparray[0], -1)
        xyz[:, 1] = np.reshape(pc_nparray[1], -1)
        xyz[:, 2] = np.reshape(pc_nparray[2], -1)
        # xyzi[:, 3] = np.reshape(pc_nparray[3], -1)

        intensity_from_velodyne = xyzi[:, 3]/255
        # print(np.amax(intensity_from_velodyne))
        # print(intensity_from_velodyne.shape)

        intensity = np.zeros((intensity_from_velodyne.shape[0], 3))
        intensity[:, 0] = np.reshape(intensity_from_velodyne[0], -1)
        intensity[:, 1] = np.reshape(intensity_from_velodyne[0], -1)
        intensity[:, 2] = np.reshape(intensity_from_velodyne[0], -1)

        PointCloud_open3d = PointCloud()
        PointCloud_open3d.colors = Vector3dVector(intensity)

        # xyzi = np.zeros((xyzi0.shape[0], 3))
        # xyzi[:, 0] = np.reshape(pc_nparray[0], -1)
        # xyzi[:, 1] = np.reshape(pc_nparray[1], -1)
        # xyzi[:, 2] = np.reshape(pc_nparray[2], -1)
        # print(xyzi.shape)

        PointCloud_open3d.points = Vector3dVector(xyz)
        write_point_cloud("%s-%i-%s.pcd" %(out_path, zeitstample, sd_token), PointCloud_open3d, write_ascii=write_ascii_BOOL)

    # Write ego-pose.
    f = open(args.out_dir+"/ego_pose_info.txt", 'w+')
    f.write("ego_pose_info for scene %s \n (with token %s) \n" % (scene_name, scene_token))
    f.write(' '.join(ego_pose_info))
    f.close()



if __name__ == '__main__':
    # Read input parameters

    parser = argparse.ArgumentParser(description='Export a scene in Wavefront point cloud format.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #  v0.1 from 0001-0061 0092-0096 0120-0139 0161-0176
    #  scene - 0061   0103   0655  0553   0757   0796  0916  1077   1094 1100

    parser.add_argument('--verbose', default=0, type=int, help='Whether to print outputs to stdout')
    write_asciiBOOL = False
    trafo_in_originBOOL = True

    parser.add_argument('--write_ascii_BOOL', default=write_asciiBOOL, type=bool, help='write in ascii or in bin')
    parser.add_argument('--trafo_in_origin_BOOL', default=trafo_in_originBOOL, type=bool,
                        help='only transform in the egoFrame or ater that tranfo in origin')

    scene_nameSTR = 'scene-1100'
    parser.add_argument('--scene', default=scene_nameSTR, type=str, help='Name of a scene, e.g. scene-0061')

    parser.add_argument('--out_dir',
                      default='/mrtstorage/users/students/yeyang/NuScenesDataset/nuscenes-visualization/nuScenes_pcd_files/%s_afterEgoTrafo_inBin' % (
                          scene_nameSTR), type=str, help='Output folder')

    # parser.add_argument('--out_dir',
    #                   default='/mrtstorage/users/students/yeyang/NuScenesDataset/nuscenes-visualization/pointclouds/%s_OriginFrame%r_writeAscii%r' % (
    #                       scene_nameSTR, trafo_in_originBOOL, write_asciiBOOL), type=str, help='Output folder')

    args = parser.parse_args()
    out_dir = os.path.expanduser(args.out_dir)
    scene_name = args.scene
    verbose = bool(args.verbose)

    out_path = osp.join(out_dir, '%s' % scene_name)
    if osp.exists(out_path):
        print('=> File {} already exists. Aborting.'.format(out_path))
        exit()
    else:
        print('=> Extracting scene {} to {}'.format(scene_name, out_path))

    # Create output folder
    if not out_dir == '' and not osp.isdir(out_dir):
        os.makedirs(out_dir)

    # Extract point-cloud for the specified scene
    nusc = NuScenes()
    scene_tokens = [s['token'] for s in nusc.scene if s['name'] == scene_name]
    assert len(scene_tokens) == 1, 'Error: Invalid scene %s' % scene_name

    export_scene_pointcloud(nusc, out_path, scene_tokens[0], scene_name, trafo_in_originBOOL, args.write_ascii_BOOL, channel='LIDAR_TOP', verbose=verbose)
