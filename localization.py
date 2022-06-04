import copy
from abc import ABC, abstractmethod
import filterpy.common
import numpy as np
from robots.robots import ConstantAccelerationRobot2D, RandomAccelerationRobot2D, ControlledRobot2D, TwoRobotSystem
from utils import Util
from utils.animator import Animator


class BaseLocalization(ABC):
    def __init__(self, robot_system: TwoRobotSystem, count=50):
        self.robot_system = robot_system
        self.count = count
        self.dt = robot_system.dt
        self.prev_r = robot_system.get_r_measurement()
        self.idx_loc = 0

        # Every measurement we get two possible locations for the robot. The initial position is not available
        # through measurements, since the algorithm makes use of change in distance.
        self.measured_positions = np.zeros((count, 2, 2))
        self.chosen_measurements = np.zeros((count, 2))
        self.chosen_measurements[0] = [None, None]
        self.measured_positions[0] = [[None, None], [None, None]]

        self.estimated_positions = np.zeros((count, 2))

    def calculate_possible_positions(self, r, v):
        dr = (r - self.prev_r) / self.dt
        # TODO: How do we update prev_r?
        # self.prev_r = np.linalg.norm(estimated_pos)
        self.prev_r = r
        s = np.linalg.norm(v)
        alpha = np.arctan2(v[1], v[0])
        theta = np.arccos(Util.clamp(dr / s, -1, 1))

        pos1 = [r * np.cos(alpha + theta), r * np.sin(alpha + theta)] + self.robot_system.anchor_robot.pos
        pos2 = [r * np.cos(alpha - theta), r * np.sin(alpha - theta)] + self.robot_system.anchor_robot.pos
        return [pos1, pos2]

    def animate_results(self, title, save=False, plot_error_figures=False):
        target_positions = self.robot_system.all_target_positions.T
        anchor_positions = self.robot_system.all_anchor_positions.T
        estimated_positions = self.estimated_positions.T

        measurements_t = self.measured_positions[:].T
        measurements_reshaped = np.stack((measurements_t[0], measurements_t[1]), axis=1)
        measurements_1 = measurements_reshaped[0]
        measurements_2 = measurements_reshaped[1]

        real_r = np.array(self.robot_system.real_r)
        measured_r = np.array(self.robot_system.measured_r)

        ani = Animator(title, self.count, self.idx_loc, anchor_positions, target_positions, estimated_positions,
                       measurements_1, measurements_2, real_r, measured_r, save, plot_error_figures)
        ani.run()

    @abstractmethod
    def run(self):
        pass


class PositionTracking(BaseLocalization):
    def __init__(self, kf, robot_system, count=50):
        super().__init__(robot_system, count)
        self.kf = kf

        init_pos = self.robot_system.target_robot.pos - self.robot_system.anchor_robot.pos
        self.estimated_positions[0] = init_pos

    def run(self):
        for i in range(1, self.count):
            self.robot_system.update()
            measured_r = self.robot_system.get_r_measurement()
            measured_v = self.robot_system.get_v_measurement()
            measurement1, measurement2 = self.calculate_possible_positions(measured_r, measured_v)
            self.measured_positions[i] = [measurement1, measurement2]

            chosen_measurement = Util.closest_to(self.estimated_positions[i-1], [measurement1, measurement2])
            self.chosen_measurements[i] = chosen_measurement

            if self.kf:
                self.kf.predict()
                self.kf.update(chosen_measurement)
                estimated_pos = [self.kf.x[0], self.kf.x[1]]
                self.estimated_positions[i] = estimated_pos
            else:
                self.estimated_positions[i] = chosen_measurement

        self.robot_system.all_anchor_positions = np.array(self.robot_system.all_anchor_positions)
        self.robot_system.all_target_positions = np.array(self.robot_system.all_target_positions)


class MotionBasedLocalization(BaseLocalization):
    def __init__(self, robot_system: TwoRobotSystem, count=50, kf=None):
        super().__init__(robot_system, count)
        self.kf = kf
        self.localized = False
        self.prev_v = robot_system.get_v_measurement()

    def run(self):
        def find_max_similarity(prev_positions, new_positions):
            max_sim = -2
            max_position = None
            for pos in new_positions:
                for prev in prev_positions:
                    sim = Util.cos_similarity(pos, prev)
                    if sim > max_sim:
                        max_sim = sim
                        max_position = pos
            return max_position

        for i in range(1, self.count):
            self.robot_system.update()
            measured_r = self.robot_system.get_r_measurement()
            measured_v = self.robot_system.get_v_measurement()
            measurement1, measurement2 = self.calculate_possible_positions(measured_r, measured_v)
            self.measured_positions[i] = [measurement1, measurement2]

            if self.localized:
                # Compare measurements to moving average of previous positions
                window_length = 5
                window = list(range(max(self.idx_loc, i-window_length), i))
                prev_pos = np.mean(self.estimated_positions[window], axis=0)
                # prev_pos = self.estimated_positions[i-1]
                closest_measurement = Util.closest_to(prev_pos, [measurement1, measurement2])
                self.chosen_measurements[i] = closest_measurement

                if self.kf:
                    self.kf.predict()
                    self.kf.update(closest_measurement)
                    closest_measurement = [self.kf.x[0], self.kf.x[1]]
                self.estimated_positions[i] = closest_measurement
            else:
                similarity = Util.cos_similarity(self.prev_v, measured_v)
                if similarity < .99 and i > 1:
                    self.localized = True
                    prev1 = self.measured_positions[i-1][0]
                    prev2 = self.measured_positions[i-1][1]

                    max_pos = find_max_similarity([prev1, prev2], [measurement1, measurement2])
                    self.estimated_positions[i] = max_pos

                    if self.kf:
                        self.kf.x = np.concatenate((max_pos, measured_v))

                    self.chosen_measurements[i] = max_pos
                    self.idx_loc = i
            self.prev_v = measured_v

        self.robot_system.all_anchor_positions = np.array(self.robot_system.all_anchor_positions)
        self.robot_system.all_target_positions = np.array(self.robot_system.all_target_positions)


def run_motion_based_localization():
    p0 = [3., 2.]
    v0 = [1, 1]
    u = [[1, 0]] * 100 + [[1, 2]] * 100
    dt = .1
    is_noisy = True
    r_std = .01
    v_std = 0
    target_ax_std = 1.5
    target_ay_std = 1
    # u = [[1, 0], [1, 0], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
    #      [1, 2], [1, 2]]
    # target = ControlledRobot2D(u, dt=dt, init_pos=p0)

    target = RandomAccelerationRobot2D(p0, v0, dt, ax_noise=target_ax_std, ay_noise=target_ay_std)
    # anchor = RandomAccelerationRobot2D([0., 0.], [-1, 1], dt, ax_noise=1, ay_noise=1.5)

    system = TwoRobotSystem(None, target, noise=is_noisy, r_std=r_std, v_std=v_std)

    kf = filterpy.common.kinematic_kf(2, 1, dt, order_by_dim=False)
    ax_var = target_ax_std ** 2
    ay_var = target_ay_std ** 2
    kf.Q = np.array([[dt ** 4 * ax_var / 4, 0, dt ** 3 * ax_var / 2, 0],
                     [0, dt ** 4 * ay_var / 4, 0, dt ** 3 * ay_var / 2],
                     [dt ** 3 * ax_var / 2, 0, dt ** 2 * ax_var, 0],
                     [0, dt ** 3 * ay_var / 2, 0, dt ** 2 * ay_var]])
    kf.P *= copy.deepcopy(kf.Q)

    # kf.R = np.array([[23424.51598114,  9877.72176729], [9877.72176729,  5403.0224787]])
    # for r_std = 0.1
    kf.R = np.array([[2912.56428837, -1857.63133772], [-1857.63133772, 10249.59328298]])

    loc = MotionBasedLocalization(system, 1000, kf)
    loc.run()

    rmse_est = Util.rmse(loc.estimated_positions[loc.idx_loc:], system.all_target_positions[loc.idx_loc:])
    trunc_rmse_est = ['%.4f' % val for val in rmse_est]
    title = f"Motion-based localization (unknown initial position), dt = {dt}, RMSE = {trunc_rmse_est}"
    if is_noisy:
        title += f", $\sigma_r={r_std}$, $\sigma_v$ = {v_std}"
    rmse_meas = Util.rmse(loc.chosen_measurements[loc.idx_loc:], loc.robot_system.all_target_positions[loc.idx_loc:])
    print(f"RMSE_est = {rmse_est}, RMSE_meas = {rmse_meas}")

    loc.animate_results(title=title, save=False, plot_error_figures=True)


def run_position_tracking():
    count = 1000
    pos0 = [3., 2.]
    v0 = [1, 1]
    dt = .1
    r_std = .01
    v_std = 0
    is_noisy = True

    ax_std = .4
    ay_std = .7

    u = [[1, 0]] * 100 + [[1, 2]] * 100
    # u = [[1, 0], [1, 0], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
    #      [1, 2], [1, 2]]

    # anchor = RandomAccelerationRobot2D([0, 0], [1, 1], dt, ax_noise=.2, ay_noise=-.1)
    # target = RandomAccelerationRobot2D(pos0, [1, 1], dt, ax_noise=ax_std, ay_noise=ay_std)
    target = ControlledRobot2D(u, dt, pos0)
    count = len(u)
    system = TwoRobotSystem(None, target, noise=is_noisy, v_std=v_std, r_std=r_std)

    kf = filterpy.common.kinematic_kf(2, 1, dt, order_by_dim=False)
    # kf.Q = filterpy.common.Q_discrete_white_noise(2, dt=dt, var=ax_std ** 2, block_size=2, order_by_dim=False)

    ax_var = ax_std**2
    ay_var = ay_std**2
    kf.Q = np.array([[dt**4 * ax_var / 4, 0, dt**3 * ax_var / 2, 0],
                     [0, dt**4 * ay_var / 4, 0, dt**3 * ay_var / 2],
                     [dt**3 * ax_var / 2, 0, dt**2 * ax_var, 0],
                     [0, dt**3 * ay_var/2, 0, dt**2 * ay_var]])
    kf.P *= copy.deepcopy(kf.Q)

    kf.x = pos0 + v0
    kf.R *= (r_std ** 2 + v_std ** 2) * 100000  # TODO: this is wrong, needs to change

    loc = PositionTracking(None, robot_system=system, count=count)
    loc.run()

    rmse_est = Util.rmse(loc.estimated_positions, loc.robot_system.all_target_positions)

    trunc_rmse_est = ['%.4f' % val for val in rmse_est]
    title = f"Position Tracking, dt = {dt}, RMSE = {trunc_rmse_est}"
    if is_noisy:
        title += f", $\sigma_r={r_std}$, $\sigma_v$ = {v_std}"
    rmse_meas = Util.rmse(loc.chosen_measurements, loc.robot_system.all_target_positions)
    print(f"RMSE_est = {rmse_est}, RMSE_meas = {rmse_meas}")

    loc.animate_results(title=title, save=False)


def determine_r_matrix():
    count = 10000
    p0 = [3., 2.]
    v0 = [1, 1]
    dt = .1
    is_noisy = True
    r_std = .1
    v_std = 0
    target_ax_std = 1.5
    target_ay_std = 1

    reps = 1000
    q = np.array([[0., 0.], [0., 0.]])
    for i in range(reps):
        target = RandomAccelerationRobot2D(p0, v0, dt, ax_noise=target_ax_std, ay_noise=target_ay_std)
        system = TwoRobotSystem(None, target, noise=is_noisy, r_std=r_std, v_std=v_std)
        loc = MotionBasedLocalization(system, count)
        loc.run()

        if loc.localized:
            error = loc.chosen_measurements[loc.idx_loc:] - system.all_target_positions[loc.idx_loc:]
            e_nx = np.mean(error[:][0]**2)
            e_nx_ny = np.mean(error[:][0] * error[:][1])
            e_ny = np.mean(error[:][1]**2)
            q += np.array([[e_nx, e_nx_ny], [e_nx_ny, e_ny]])
        else:
            reps -= 1
    q /= reps
    print(reps)
    return q


if __name__ == '__main__':
    run_motion_based_localization()
    # run_position_tracking()
    # print(determine_e_matrix())
