from __future__ import print_function

import sys
import numpy
import yaml
import transformations

def eprint(*args, **kwargs):
    """
    print to stderr instead of stdout
    :param args: same as that in print
    :param kwargs: same as that in print
    :return:
    """
    print(*args, file=sys.stderr, **kwargs)


def euroc_pose_2_transform(euroc_pose):
    """
    Convert from EuRoC pose format to transformation matrix
    :param euroc_pose: tx,ty,tz,qw,qx,qy,qz
    :return: equivalent transformation matrix
    """
    T = transformations.quaternion_matrix(euroc_pose[3:7])
    T[:3, 3] = euroc_pose[:3]
    return T

def tum_2_transform(tum_pose):
    """
    Convert from TUM-RGBD pose format to transformation matrix
    :param tum_pose: tum pose tx ty tz qx qy qz qw
    :return: equivalent transformation matrix
    """
    # convert from TUM(qx,qy,qz,qw) to EuRoC(qw,qx,qy,qz) format and call euroc_pose_2_transform
    return euroc_pose_2_transform(tum_pose[0:3] + [tum_pose[6]] + tum_pose[3:6])

def transformation_2_tum_pose(T):
    """
    Convert from transformation matrix to TUM-RGBD dataset pose
    :param T: transformation matrix
    :return: tum pose: tx ty tz qx qy qz qw
    """
    q = transformations.quaternion_from_matrix(T)
    return [
        T[0, 3], T[1, 3],T[2, 3],
        q[1], q[2], q[3], q[0] # x,y,z,w
    ]

def get_T_BS(config_file):
    """
    Gets transformation from body frame (imu) to sensor frame (camera)
    :param config_file: the sensor.yaml file containing sensor to body transformation
    :return: the transformation matrix if successful, None if not
    """
    config = None
    with open(config_file, 'r') as stream:
        config = yaml.load(stream)

    if not config:
        eprint("bad config file")
        return None

    if (not 'T_BS' in config) or (not 'data' in config['T_BS']):
        eprint("required keys not in the config file")
        return None
    T_BS = numpy.asarray(config['T_BS']['data'])
    T_BS = numpy.resize(T_BS, (4, 4))

    return T_BS

def convert_groundtruth(groundtruth):
    """
    Convert groundtruth from EuRoC format to TUM format
    :param groundtruth: a dictionary of groundtruth data in the EuRoC format timestamp(ns),tx,ty,tz,qw,qx,qy,qz)
    :return: equivalent dictionary in TUM-RGBD format timestamp(s) tx ty tz qx qy qz qw
    """
    assert(type(groundtruth) == dict)
    return {
        (i*1e-9): transformation_2_tum_pose( euroc_pose_2_transform(groundtruth[i][:7]) )
        for i in groundtruth
    }

def convert_traj_to_body_frame(trajectory, T_BS):
    """
    Transform trajectory to body frame from sensor frame
    :param trajectory tx ty tz qx qy qz qw
    :param T_BS camera to imu transformation
    :return:
    """
    assert(type(trajectory) == dict)
    return {
        i: transformation_2_tum_pose( numpy.dot(T_BS, tum_2_transform(trajectory[i][:7])) )
        for i in trajectory
    }
