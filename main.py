import filterpy.common

from position_tracking import PositionTracking
from robots.constant_acceleration_robot import ConstantAccelerationRobot2D

# The higher the velocity, the more noise in the measurements? (a = [2,0])
# Not true, a = [2,2] doesn't have a lot of noise.
# This might also link with lower dt corresponding to more noise.

# a = [0, 0] has a high noise???
from robots.controlled_robot import ControlledRobot

if __name__ == '__main__':
    pos0 = [3., 2.]
    v0 = [1., 3.]
    acc = [.5, .50]
    # Lower dt -> more noise (why? maybe because velocity grows slower?)
    dt = 3
    r_std = 0.01
    v_std = .01
    rob = ConstantAccelerationRobot2D(
        init_pos=pos0, init_vel=v0, accel=acc, dt=dt, r_std=r_std, v_std=v_std, noise=False)

    # Constant acceleration model (2 dimensional, 2nd order)
    kf = filterpy.common.kinematic_kf(2, 2, dt, order_by_dim=False)
    # x = [x, y, vx, vy, ax, ay]
    kf.x = pos0 + v0 + acc
    kf.R *= (r_std**2 + v_std**2)
    # Low, bcs we are confident in initial position
    kf.P *= 0.001
    kf.Q *= 0

    pt = PositionTracking(kf, rob, count=3)
    pt.run()
