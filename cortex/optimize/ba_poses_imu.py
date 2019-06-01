from util import compiler
from ctypes import *


def bundle_adjuster(poses):
    ba = CDLL('./optimize/bundle_adjuster.so')
    setup = compiler.cfunc('setup', ba, c_void_p)
    release = compiler.cfunc('release', ba, None, ('adapter_ptr', c_void_p, 1))
    add_poses = compiler.cfunc('add_poses', ba, c_int,
                       ('adapter_ptr', c_void_p, 1),
                       ('poses', POINTER(c_double), 1),
                       ('number_poses', c_int, 1))
    add_imu_residuals = compiler.cfunc('add_imu_residuals', ba, c_int, ('adapter_ptr', c_void_p, 1))
    optimize = compiler.cfunc('optimize', ba, c_int, ('adapter_ptr', c_void_p, 1))

    poses_tmp = compiler.matrix_to_ctype_array(poses)
    p = setup()
    status = add_poses(p, poses_tmp, len(poses))
    status = add_imu_residuals(p)
    status = optimize(p)
    release(p)


def optimize_poses_imu(poses):
    compiler.compile('./optimize', 'bundle_adjuster')
    bundle_adjuster(poses)
