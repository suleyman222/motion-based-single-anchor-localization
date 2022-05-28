from abc import ABC, abstractmethod

import matplotlib.animation
import numpy as np
from matplotlib import pyplot as plt

from robots.accelerating_robots import ConstantAccelerationRobot2D, RandomAccelerationRobot2D
from robots.base_robot import BaseRobot2D, TwoRobotSystem
from robots.controlled_robot import ControlledRobot2D
from utils import Util


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

        pos1 = [r * np.cos(alpha + theta), r * np.sin(alpha + theta)]
        pos2 = [r * np.cos(alpha - theta), r * np.sin(alpha - theta)]
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

    def calculate_position(self, prev_pos, r, v):
        return Util.closest_to(prev_pos, self.calculate_possible_positions(r, v))


class MotionBasedLocalization(BaseLocalization):
    def __init__(self, robot_system: TwoRobotSystem, count=50):
        super().__init__(robot_system, count)
        self.localized = False
        self.idx_localized = None
        self.prev_v = [None, None]

        # Every measurement we get two possible locations for the robot. The initial position is not available
        # through measurements, since the algorithm makes use of change in distance.
        self.measured_positions = np.array([([None, None], [None, None])] * (self.count + 1))
        self.chosen_positions = np.array([[None, None]] * (self.count + 1))

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
                measured_pos = Util.closest_to(self.chosen_positions[i], [pos1, pos2])
                self.chosen_positions[i+1] = measured_pos
            else:
                similarity = Util.cos_similarity(self.prev_v, measured_v)
                if similarity < .99:
                    self.localized = True
                    prev1 = self.measured_positions[i][0]
                    prev2 = self.measured_positions[i][1]

                    max_pos = find_max_similarity([prev1, prev2], [pos1, pos2])
                    self.chosen_positions[i + 1] = max_pos
                    self.idx_localized = i + 1
            self.prev_v = measured_v

        # Calculate the positions of the robot before it was precisely localized
        if self.idx_localized:
            for i in reversed(range(self.idx_localized)):
                if i == 0:
                    continue
                self.chosen_positions[i] = Util.closest_to(self.chosen_positions[i + 1], self.measured_positions[i])

    def plot_results(self, show_all_measurements=False):
        rel_positions = [t_pos - a_pos for (t_pos, a_pos) in zip(self.robot_system.target_robot.all_positions, self.robot_system.anchor_robot.all_positions)]
        all_pos = np.array(rel_positions)
        alt1 = np.array([possible_position[0] for possible_position in self.measured_positions])
        alt2 = np.array([possible_position[1] for possible_position in self.measured_positions])
        chosen_positions = np.array(self.chosen_positions)

        if not self.localized:
            plt.title("Could not localize")
        else:
            rmse = Util.rmse(chosen_positions[1:], all_pos[1:])
            trunc_rmse = ['%.4f' % val for val in rmse]
            plt.title(f"RMSE = {trunc_rmse}")
            plt.plot(chosen_positions[self.idx_localized][0], chosen_positions[self.idx_localized][1], 'r+',
                     ms=10, label="First precisely located position")

        plt.plot(all_pos[:, 0], all_pos[:, 1], label="Actual path")
        plt.plot(chosen_positions[:, 0], chosen_positions[:, 1], label="Found path")

        if show_all_measurements:
            plt.plot(alt1[:, 0], alt1[:, 1], 'g--', label="Measurement 1")
            plt.plot(alt2[:, 0], alt2[:, 1], 'y--', label="Measurement 2")

        plt.legend()
        plt.show()

    def animate_results(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line_real, = ax.plot([], [], 'b-', ms=10, label="Actual path")
        line_chosen, = ax.plot([], [], 'r-', ms=10, label="Found path")
        line_measured1, = ax.plot([], [], 'g--', ms=10, label="Measurement 1")
        line_measured2, = ax.plot([], [], 'y--', ms=10, label="Measurement 2")

        # TODO: Fix these real positions
        real_positions = np.array(self.robot_system.target_robot.all_positions)
        real_x = real_positions[:, 0]
        real_y = real_positions[:, 1]

        chosen_x = self.chosen_positions[1:, 0]
        chosen_y = self.chosen_positions[1:, 1]
        ax.set_xlim(np.min(chosen_x), np.max(chosen_x))
        ax.set_ylim(np.min(chosen_y), np.max(chosen_y))
        ax.legend()

        def animate(frame):
            line_real.set_xdata(real_x[:frame])
            line_real.set_ydata(real_y[:frame])

            if self.localized:
                line_chosen.set_xdata(chosen_x[self.idx_localized:frame])
                line_chosen.set_ydata(chosen_y[self.idx_localized:frame])

            if not self.localized or frame < self.idx_localized:
                line_measured1.set_xdata(self.measured_positions[:frame, 0, 0])
                line_measured1.set_ydata(self.measured_positions[:frame, 0, 1])
                line_measured2.set_xdata(self.measured_positions[:frame, 1, 0])
                line_measured2.set_ydata(self.measured_positions[:frame, 1, 1])

            return line_real, line_chosen, line_measured1, line_measured2

        ani = matplotlib.animation.FuncAnimation(fig, animate, frames=self.count, interval=10,
                                                 save_count=self.count, blit=True)
        ani.save('ani.gif', 'pillow')


def run_static_anchor():
    p0 = [3., 2.]
    u = [[1, 0]] * 100 + [[1, 2]] * 100
    # u = [[1, 0], [1, 0], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
    #      [1, 2], [1, 2]]
    # cr = ControlledRobot2D(u, p0, dt=.1, noise=True, r_std=0.001, v_std=0.001)
    # cr = ConstantAccelerationRobot2D(p0, [.1, .1], [.1, .1], dt=.1)
    cr = RandomAccelerationRobot2D(p0, [1, 1], .1, ax_noise=10, ay_noise=1, noise=False, r_std=.0001, v_std=0)
    cr_anchor = RandomAccelerationRobot2D([0., 0.], [-1, 1], .1, ax_noise=1, ay_noise=10, noise=False, r_std=.0001, v_std=0)

    system = TwoRobotSystem(cr_anchor, cr)
    loc = MotionBasedLocalization(system, len(u))
    loc.run()
    loc.plot_results(show_all_measurements=False)
    loc.animate_results()


def run_mobile_anchor():
    count = 100
    dt = .1
    anchor = RandomAccelerationRobot2D([0, 0], [1, 1], dt, ax_noise=-5, ay_noise=-1)
    target = RandomAccelerationRobot2D([3, 2], [1, 1], dt, ax_noise=2, ay_noise=7)
    system = TwoRobotSystem(None, target)

    loc = PositionTracking(None, robot_system=system, count=count)
    loc.run()

    fig, axs = plt.subplots(2)

    # Robot paths
    axs[0].plot([pos[0] for pos in anchor.all_positions], [pos[1] for pos in anchor.all_positions])
    axs[0].plot([pos[0] for pos in target.all_positions], [pos[1] for pos in target.all_positions])

    # Location of target relative to anchor robot
    anchor_pos = np.array(anchor.all_positions)
    target_pos = np.array(target.all_positions)
    relative_pos = target_pos - anchor_pos
    axs[1].plot(relative_pos[:, 0], relative_pos[:, 1])
    axs[1].plot(loc.estimated_positions[:, 0], loc.estimated_positions[:, 1])

    plt.show()


if __name__ == '__main__':
    run_static_anchor()
    # run_mobile_anchor()
