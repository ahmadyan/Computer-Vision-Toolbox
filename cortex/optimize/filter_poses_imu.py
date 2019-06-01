from util import compiler
from ctypes import *
import numpy as np

def filter_imu(timestamp, gyro, accel, magnet):
    ekf = CDLL('./optimize/ekf_smoother.so')
    setup = compiler.cfunc('setup', ekf, c_void_p, ('gyro', compiler.c_array(c_double), 1))
    release = compiler.cfunc('release', ekf, None, ('adapter_ptr', c_void_p, 1))
    attitude = compiler.cfunc('attitude', ekf, c_int, ('ptr', c_void_p, 1), ('state', POINTER(c_double), 1))
    update = compiler.cfunc('update', ekf, c_int,
                             ('ptr', c_void_p, 1),
                             ('gyro', compiler.c_array(c_double), 1),
                             ('acc', compiler.c_array(c_double), 1),
                             ('magnet', compiler.c_array(c_double), 1),
                             ('timestamp', c_double, 1))

    # Initial covariance values for IMU signal, See Bar-Shalom, p - 499, todo:needs tuning
    # Accel, Gyro, Magnet and process covariance
    cov = [10, 0.05, 10, 0.1]

    number_measurements = len(timestamp)
    state_dimension = 4
    state = np.zeros((state_dimension, 1))
    att = np.zeros((number_measurements, state_dimension))

    p = setup(cov)
    for i in range(number_measurements):
        status = update(p, gyro[i, :].tolist(), accel[i, :].tolist(), magnet[i,:].tolist(), timestamp[i])
        status = attitude(p, state.ctypes.data_as(POINTER(c_double)))
        att[[i],:] = state.T
    release(p)
    return att
