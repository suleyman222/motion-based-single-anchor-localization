import filterpy.common
import numpy as np

from localization import PositionTracking
from robots.accelerating_robots import ConstantAccelerationRobot2D, RandomAccelerationRobot2D
from utils import Util


# The higher the velocity, the more noise in the measurements? (a = [2,0])
# Not true, a = [2,2] doesn't have a lot of noise.
# This might also link with lower dt corresponding to more noise.
# Has something to do with the amount acceleration compared to velocity (relatively smaller a -> more noise).
# a = [0, 0] has a high noise???

# Seems to be a bug somewhere, the measurements are not perfect without noise. Maybe it's normal: a straight path
# has a lot less deviation in the measurement compared to a curving path.

# Explain kind of lag in changing velocity direction (seems to stay the same while changing dt)
# Explain more noise in later part of curving path
# Lower dt -> more noise (why? maybe because velocity grows slower?)

# In general, noise seems to get worse the longer the path goes on. Why? Even though measurement noise is the same, the
# increased velocity probably causes the high noise in the later stages of the path.

# Main problem:
# The changes in position need to be small every timestep, but this means that noise in the distance and velocity
# measurements hit extremely hard.


def random_acc_run():
    is_noisy = True
    pos0 = [3., 2.]
    v0 = [1, 1]
    dt = .1
    r_std = .0005
    v_std = .0005
    acc_std = 5

    rand_kf = filterpy.common.kinematic_kf(2, 1, dt, order_by_dim=False)
    # Noise combined (?!?!) with smaller acc_std (relative to v) results in awful measurements

    rand_kf.Q = filterpy.common.Q_discrete_white_noise(2, dt=dt, var=acc_std**2,
                                                       block_size=2, order_by_dim=False)
    rand_kf.P *= 0

    rand_kf.x = pos0 + v0
    rand_kf.R *= (r_std ** 2 + v_std ** 2) * 1000000000  # TODO: this is wrong, needs to change
    # rand_kf.B = np.array([[0.5*dt**2, 0], [0, 0.5*dt**2], [dt, 0], [0, dt]])
    # print(rand_kf)

    title = f"Noiseless measurements, $\sigma_a = {acc_std}$, dt = {dt}" if not is_noisy \
        else f" $\sigma_r={r_std}$, $\sigma_v$ = {v_std}, $\sigma_a = {acc_std}$, dt = {dt}"

    rand_rob = RandomAccelerationRobot2D(
        init_pos=pos0, init_vel=v0, dt=dt, r_std=r_std, v_std=v_std, noise=is_noisy, ax_noise=acc_std, ay_noise=acc_std)
    rand_pt = PositionTracking(rand_kf, rand_rob, 1000)

    measured_positions, estimated_positions = rand_pt.run()
    real_positions = np.array(rand_rob.all_positions)
    rmse_measurement = Util.rmse(measured_positions, real_positions)
    rmse_estimate = Util.rmse(estimated_positions, real_positions)

    Util.plot_path(real_positions, measured_positions, estimated_positions, title)

    return rmse_measurement, rmse_estimate


def constant_acc_run():
    is_noisy = True
    pos0 = [3., 2.]
    v0 = [1, 1]
    acc = [.5, .5]
    dt = .5
    r_std = .001
    v_std = .001
    rob = ConstantAccelerationRobot2D(
        init_pos=pos0, init_vel=v0, accel=acc, dt=dt, r_std=r_std, v_std=v_std, noise=is_noisy)

    # Constant acceleration model (2 dimensional, 2nd order)
    kf = filterpy.common.kinematic_kf(2, 2, dt, order_by_dim=False)

    # xEst = [x, y, vx, vy, ax, ay]
    kf.x = pos0 + v0 + acc

    kf.R *= (r_std**2 + v_std**2)

    # Low, bcs we are confident in initial position
    kf.P *= .00001
    kf.Q *= 0

    title = f"Noiseless measurements, dt = {dt}" if not is_noisy else f" $\sigma_r={r_std}$, $\sigma_v = {v_std}$, dt = {dt}"
    pt = PositionTracking(kf, rob, count=100)

    measured_positions, estimated_positions = pt.run()
    real_positions = np.array(rob.all_positions)

    Util.plot_path(real_positions, measured_positions, estimated_positions, title)

    rmse_measurements = Util.rmse(measured_positions, real_positions)
    print(rmse_measurements)



if __name__ == '__main__':
    # runs = 1000
    # avg = 0
    # for _ in range(runs):
    #     avg += random_acc_run()
    # avg /= runs
    # print(avg)

    rmse_measurements, rmse_estimates = random_acc_run()
    print(f"RMSE of measurements: {rmse_measurements}")
    print(f"RMSE of estimate: {rmse_estimates}")

    # constant_acc_run()

# Random acceleration model:
# a_std = 1, dt=1 -> RMSE = [0.66135563 0.64496473]
# a_std = .5, dt=1 -> RMSE = [0.28907013 0.35538076]
# a_std = .3, dt=1 -> RMSE = [0.21152789 0.21231493]
