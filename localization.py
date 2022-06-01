import copy
from abc import ABC, abstractmethod

import filterpy.common
import matplotlib.animation
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from robots.robots import ConstantAccelerationRobot2D, RandomAccelerationRobot2D, ControlledRobot2D, TwoRobotSystem
from utils import Util
from utils.animator import Animator


class BaseLocalization(ABC):
    def __init__(self, robot_system: TwoRobotSystem, count=50):
        self.robot_system = robot_system
        self.count = count
        self.dt = robot_system.dt
        self.prev_r = np.linalg.norm(robot_system.target_robot.pos - robot_system.anchor_robot.pos)

        self.measured_positions = np.zeros((count + 1, 2))
        self.estimated_positions = np.zeros((count + 1, 2))

    def calculate_possible_positions(self, r, v):
        dr = (r - self.prev_r) / self.dt
        self.prev_r = r
        s = np.linalg.norm(v)
        alpha = np.arctan2(v[1], v[0])
        theta = np.arccos(Util.clamp(dr / s, -1, 1))

        pos1 = [r * np.cos(alpha + theta), r * np.sin(alpha + theta)] + self.robot_system.anchor_robot.pos
        pos2 = [r * np.cos(alpha - theta), r * np.sin(alpha - theta)] + self.robot_system.anchor_robot.pos
        return [pos1, pos2]

    @abstractmethod
    def run(self):
        pass


class PositionTracking(BaseLocalization):
    def __init__(self, kf, robot_system, count=50):
        super().__init__(robot_system, count)
        self.kf = kf

        init_pos = self.robot_system.target_robot.pos - self.robot_system.anchor_robot.pos
        self.estimated_positions[0] = init_pos
        self.measured_positions[0] = init_pos

    def run(self):
        for i in range(1, self.count + 1):
            self.robot_system.update()
            measured_r, measured_v = self.robot_system.get_measurement()
            measured_pos = self.calculate_position(self.estimated_positions[i-1], measured_r, measured_v)
            self.measured_positions[i] = measured_pos

            if self.kf:
                self.kf.predict()
                self.kf.update(measured_pos)
                estimated_pos = [self.kf.x[0], self.kf.x[1]]
                self.estimated_positions[i] = estimated_pos
            else:
                self.estimated_positions[i] = measured_pos

            # TODO: How do we update prev_r?
            # self.prev_r = np.linalg.norm(estimated_pos)
            self.prev_r = measured_r

        self.robot_system.all_anchor_positions = np.array(self.robot_system.all_anchor_positions)
        self.robot_system.all_target_positions = np.array(self.robot_system.all_target_positions)

    def calculate_position(self, prev_pos, r, v):
        return Util.closest_to(prev_pos, self.calculate_possible_positions(r, v))

    def animate_results(self, save, title):
        ani = Animator(self, title, save)
        ani.run()


class MotionBasedLocalization(BaseLocalization):
    def __init__(self, robot_system: TwoRobotSystem, count=50):
        super().__init__(robot_system, count)
        self.localized = False
        self.idx_localized = None
        self.prev_v = [None, None]

        # Used for plotting
        self.is_manual = False

        # Every measurement we get two possible locations for the robot. The initial position is not available
        # through measurements, since the algorithm makes use of change in distance.
        self.measured_positions = np.array([([None, None], [None, None])] * (self.count + 1))
        self.estimated_positions = np.array([[None, None]] * (self.count + 1))

        # TODO: noise and filtering

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

        for i in range(self.count):
            self.robot_system.update()
            measured_r, measured_v = self.robot_system.get_measurement()
            [pos1, pos2] = self.calculate_possible_positions(measured_r, measured_v)
            self.measured_positions[i + 1] = (pos1, pos2)

            if i == 0:
                self.prev_v = measured_v
                continue

            if self.localized:
                measured_pos = Util.closest_to(self.estimated_positions[i], [pos1, pos2])
                self.estimated_positions[i+1] = measured_pos
            else:
                similarity = Util.cos_similarity(self.prev_v, measured_v)
                if similarity < .99:
                    self.localized = True
                    prev1 = self.measured_positions[i][0]
                    prev2 = self.measured_positions[i][1]

                    max_pos = find_max_similarity([prev1, prev2], [pos1, pos2])
                    self.estimated_positions[i + 1] = max_pos
                    self.idx_localized = i + 1
            self.prev_v = measured_v

        # Calculate the positions of the robot before it was precisely localized
        if self.idx_localized:
            for i in reversed(range(self.idx_localized)):
                if i == 0:
                    continue
                self.estimated_positions[i] = Util.closest_to(self.estimated_positions[i + 1], self.measured_positions[i])

        self.robot_system.all_anchor_positions = np.array(self.robot_system.all_anchor_positions)
        self.robot_system.all_target_positions = np.array(self.robot_system.all_target_positions)

    def plot_results(self, show_all_measurements=False):
        all_pos = np.array(self.robot_system.all_target_positions)
        alt1 = np.array([possible_position[0] for possible_position in self.measured_positions])
        alt2 = np.array([possible_position[1] for possible_position in self.measured_positions])
        chosen_positions = np.array(self.estimated_positions)

        if not self.localized:
            plt.title("Could not localize")
        else:
            rmse = Util.rmse(chosen_positions[1:], all_pos[1:])
            trunc_rmse = ['%.4f' % val for val in rmse]
            plt.title(f"Motion-based localization (unknown initial position), dt = {self.dt}, RMSE = {trunc_rmse}")
            plt.plot(chosen_positions[self.idx_localized][0], chosen_positions[self.idx_localized][1], 'r+',
                     ms=10, label="First precisely located position")

        if show_all_measurements:
            plt.plot(alt1[:, 0], alt1[:, 1], 'g--', label="Measurement 1")
            plt.plot(alt2[:, 0], alt2[:, 1], 'y--', label="Measurement 2")

        plt.plot(all_pos[:, 0], all_pos[:, 1], 'b-', label="Actual path")
        plt.plot(chosen_positions[:, 0], chosen_positions[:, 1], 'r-', label="Found path")

        plt.legend()
        plt.show()

    def animate_results(self, save, title):
        # TODO: Refactor this into Animator (make PositionTracking show both measurements)
        fig, axs = plt.subplots(1, 3)
        # ax = fig.add_subplot(111)
        axs[0].set_title(title)
        line_anchor, = axs[0].plot([], [], 'm-', ms=10, label="Anchor robot path")
        line_actual_target, = axs[0].plot([], [], 'b-', ms=10, label="Actual target path")
        line_measured1, = axs[0].plot([], [], 'g--', ms=10, label="Measurement 1")
        line_measured2, = axs[0].plot([], [], 'y--', ms=10, label="Measurement 2")
        line_chosen, = axs[0].plot([], [], 'r-', ms=10, label="Found path")

        anchor_x = self.robot_system.all_anchor_positions[:, 0]
        anchor_y = self.robot_system.all_anchor_positions[:, 1]

        target_x = self.robot_system.all_target_positions[:, 0]
        target_y = self.robot_system.all_target_positions[:, 1]

        chosen_x = self.estimated_positions[1:, 0]
        chosen_y = self.estimated_positions[1:, 1]

        measured1_x = self.measured_positions[1:, 0, 0]
        measured1_y = self.measured_positions[1:, 0, 1]
        measured2_x = self.measured_positions[1:, 1, 0]
        measured2_y = self.measured_positions[1:, 1, 1]

        axs[0].set_xlim(np.min([measured1_x, measured2_x]), np.max([measured1_x, measured2_x]))
        axs[0].set_ylim(np.min([measured1_y, measured2_y]), np.max([measured1_y, measured2_y]))
        axs[0].legend()

        # Slider
        axamp = plt.axes([0.25, .03, 0.50, 0.02])
        samp = Slider(axamp, 'Timestep', 0, self.count, valinit=0, valstep=1)

        def animate(frame):
            if self.is_manual:
                return line_anchor, line_actual_target, line_chosen, line_measured1, line_measured2

            # Calls update due to change
            samp.set_val(frame)
            self.is_manual = False
            return line_anchor, line_actual_target, line_chosen, line_measured1, line_measured2

        def update_slider(val):
            self.is_manual = True
            update(val)

        def update(val):
            line_anchor.set_data(anchor_x[:val], anchor_y[:val])
            line_actual_target.set_data(target_x[:val], target_y[:val])
            if self.localized:
                line_chosen.set_data(chosen_x[self.idx_localized:val], chosen_y[self.idx_localized:val])
            # if not self.localized or val < self.idx_localized:
            line_measured1.set_data(self.measured_positions[:val, 0, 0], self.measured_positions[:val, 0, 1])
            line_measured2.set_data(self.measured_positions[:val, 1, 0], self.measured_positions[:val, 1, 1])
            fig.canvas.draw_idle()

        def on_click(event):
            # Check where the click happened
            (xm, ym), (xM, yM) = samp.label.clipbox.get_points()
            if xm < event.x < xM and ym < event.y < yM:
                # Event happened within the slider, ignore since it is handled in update_slider
                return
            else:
                # user clicked somewhere else on canvas = unpause
                self.is_manual = False

        # call update function on slider value change
        samp.on_changed(update_slider)
        fig.canvas.mpl_connect('button_press_event', on_click)

        ani = matplotlib.animation.FuncAnimation(fig, animate, frames=self.count, interval=50, save_count=self.count+1)

        if save:
            ani.save('ani.gif', 'pillow')
        self.robot_system.plot_distances(axs[1], axs[2])
        plt.show()


def run_motion_based_localization():
    p0 = [3., 2.]
    u = [[1, 0]] * 100 + [[1, 2]] * 100
    dt = .1
    # u = [[1, 0], [1, 0], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
    #      [1, 2], [1, 2]]
    # cr = ControlledRobot2D(u, dt=dt, init_pos=p0)
    # cr = ConstantAccelerationRobot2D(p0, [.1, .1], [.1, .1], dt=dt)
    cr = RandomAccelerationRobot2D(p0, [1, 1], dt, ax_noise=1.5, ay_noise=1)
    cr_anchor = RandomAccelerationRobot2D([0., 0.], [-1, 1], dt, ax_noise=1, ay_noise=1.5)

    system = TwoRobotSystem(None, cr, noise=True, r_std=.001, v_std=0)
    loc = MotionBasedLocalization(system, len(u))
    loc.run()
    loc.plot_results(show_all_measurements=True)

    rmse_est = Util.rmse(loc.estimated_positions[1:], loc.robot_system.all_target_positions[1:])
    trunc_rmse_est = ['%.4f' % val for val in rmse_est]
    title = f"Motion-based localization (unknown initial position), dt = {dt}, RMSE = {trunc_rmse_est}"

    loc.animate_results(save=False, title=title)


def run_position_tracking():
    count = 1000
    pos0 = [3., 2.]
    v0 = [1, 1]
    dt = .1
    r_std = 0
    v_std = .1
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
    rmse_meas = Util.rmse(loc.measured_positions, loc.robot_system.all_target_positions)
    print(f"RMSE_est = {rmse_est}, RMSE_meas = {rmse_meas}")

    loc.animate_results(save=False, title=title)


if __name__ == '__main__':
    run_motion_based_localization()
    # run_position_tracking()
