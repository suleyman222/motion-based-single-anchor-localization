import copy
import filterpy.kalman
import numpy as np

from localization_algorithm import MotionBasedLocalization
from robots.robots import RandomAccelerationRobot2D
from robots.two_robot_system import TwoRobotSystem
from utils import util

P0 = [3., 2.]
V0 = [1, 1]
DT = .5
IS_NOISY = True
USE_RSSI_MODEL = True
R_STD = 1.
V_STD = 0.1
TARGET_AX_STD = 1.5
TARGET_AY_STD = 1.
COUNT = 400
# COUNT = len(INPUT_PATH)

# INPUT_PATH = [[1., 0.]] * 50 + [[1., 2.]] * 20 + [[-2, 1]] * 50 + [[-1, -1]] * 100 + [[0, 2]] * 50
# Shows that noise increases the further the robot gets
# u = [[1., 0.]] * 25 + [[1., 2.]] * 25 + [[2, 1]] * 100 + [[0, 1]] * 1000 + [[1, 0]] * 100 +  [[0, -1]] * 1000

# Shows that dr has a lot more impact on bigger r
# u = [[1., 0.]] * 25 + [[1., 2.]] * 25 + [[1., -2.]] * 25 + [[1., 0.]] * 1000
# u = [[1., -2.]] * 10 + [[-1., -2.]] * 10 + [[0., -1.]] * 1000


def run_motion_based_localization():
    # target = ControlledRobot2D(u, dt=dt, init_pos=p0, init_vel=u[0])
    target = RandomAccelerationRobot2D(P0, V0, DT, ax_noise=TARGET_AX_STD, ay_noise=TARGET_AY_STD)
    # anchor = RandomAccelerationRobot2D([0., 0.], [-1, 1], dt, ax_noise=1, ay_noise=1.5)

    kf = filterpy.kalman.KalmanFilter(4, 4)
    kf.F = np.array([[1.,  0.,  DT, 0.],
                     [0.,  1.,  0.,  DT],
                     [0.,  0.,  1.,  0.],
                     [0.,  0.,  0.,  1.]])

    ax_var = TARGET_AX_STD ** 2
    ay_var = TARGET_AY_STD ** 2
    kf.Q = np.array([[DT ** 4 * ax_var / 4, 0, DT ** 3 * ax_var / 2, 0],
                     [0, DT ** 4 * ay_var / 4, 0, DT ** 3 * ay_var / 2],
                     [DT ** 3 * ax_var / 2, 0, DT ** 2 * ax_var, 0],
                     [0, DT ** 3 * ay_var / 2, 0, DT ** 2 * ay_var]])

    kf.R = np.array([[21.62548567,  0.99567984, 0, 0],
                     [0.99567984, 21.36388223, 0, 0],
                     [0, 0, V_STD, 0],
                     [0, 0, 0, V_STD]])
    kf.P = copy.deepcopy(kf.R)
    kf.H = np.eye(4)

    system = TwoRobotSystem(None, target, is_noisy=IS_NOISY, r_std=R_STD, v_std=V_STD, rssi_noise=USE_RSSI_MODEL)
    loc = MotionBasedLocalization(system, COUNT, kf)
    loc.run()

    rmse_est = util.rmse(loc.estimated_positions[loc.idx_loc:], system.all_target_positions[loc.idx_loc:])
    trunc_rmse_est = ['%.4f' % val for val in rmse_est]
    title = f"Motion-based localization (unknown initial position), dt = {DT}, RMSE = {trunc_rmse_est}"
    if IS_NOISY:
        if USE_RSSI_MODEL:
            title += f", RSSI noise"
        else:
            title += f", $\sigma_r={R_STD}$, $\sigma_v$ = {V_STD}"
    rmse_meas = util.rmse(loc.chosen_measurements[loc.idx_loc:], loc.robot_system.all_target_positions[loc.idx_loc:])
    print(f"RMSE_est = {rmse_est}, RMSE_meas = {rmse_meas}")

    loc.animate_results(title=title, save=False, plot_error_figures=True)


if __name__ == '__main__':
    run_motion_based_localization()
