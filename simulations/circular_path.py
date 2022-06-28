import copy

import filterpy.kalman
import numpy as np

from localization_algorithm import MotionBasedLocalization
from robots.robots import RotatingRobot2D
from robots.two_robot_system import TwoRobotSystem
from utils import util

P0 = [-9, -5]
COUNT = 400
DT = .5
IS_NOISY = True
USE_RSSI_MODEL = True
R_STD = 1.
V_STD = 0.1


def run_rotating_robot():
    # Instantiate Kalman filter
    kf = filterpy.kalman.KalmanFilter(4, 4)
    kf.F = np.array([[1., 0., DT, 0.],
                     [0., 1., 0., DT],
                     [0.,  0.,  1.,  0.],
                     [0.,  0.,  0.,  1.]])
    sigma_q = .1 ** 2
    kf.Q = np.array([[DT ** 4 * sigma_q / 4, 0, DT ** 3 * sigma_q / 2, 0],
                     [0, DT ** 4 * sigma_q / 4, 0, DT ** 3 * sigma_q / 2],
                     [DT ** 3 * sigma_q / 2, 0, DT ** 2 * sigma_q, 0],
                     [0, DT ** 3 * sigma_q / 2, 0, DT ** 2 * sigma_q]])

    # Determined R empirically by running the simulation with noise
    kf.R = np.array([[38.64475783, -0.72235203, 0, 0],
                     [-0.72235203, 39.07971159, 0, 0],
                     [0, 0, V_STD, 0],
                     [0, 0, 0, V_STD]])
    kf.P = copy.deepcopy(kf.R)
    kf.H = np.eye(4)

    # Instantiate and run robot system
    target = RotatingRobot2D(init_pos=P0, dt=DT)
    system = TwoRobotSystem(None, target, is_noisy=IS_NOISY, r_std=R_STD, v_std=V_STD, rssi_noise=USE_RSSI_MODEL)
    loc = MotionBasedLocalization(system, COUNT, kf, known_initial_pos=False)
    loc.run()

    rmse_est = util.rmse(loc.estimated_positions[loc.idx_loc:], system.all_target_positions[loc.idx_loc:])
    rmse_conv = util.rmse(loc.estimated_positions[150:], system.all_target_positions[150:])
    trunc_rmse_est = ['%.4f' % val for val in rmse_est]
    title = f"Motion-based localization (unknown initial position), dt = {DT}, RMSE = {trunc_rmse_est}"
    if IS_NOISY:
        if USE_RSSI_MODEL:
            title += f", RSSI noise"
        else:
            title += f", $\sigma_r={R_STD}$, $\sigma_v$ = {V_STD}"
    rmse_meas = util.rmse(loc.chosen_measurements[loc.idx_loc:], loc.robot_system.all_target_positions[loc.idx_loc:])
    print(f"RMSE_est = {rmse_est}, RMSE_meas = {rmse_meas}, RMSE_conv = {rmse_conv}")

    loc.animate_results(title=title, save=False, plot_error_figures=True)
    return rmse_conv


if __name__ == '__main__':
    run_rotating_robot()
