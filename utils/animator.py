import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from itertools import chain
import matplotlib.animation


class Animator:
    def __init__(self, loc, title: str, save=False, plot_error_figures=False):
        self.plot_error_figures = plot_error_figures
        self.save = save
        self.loc = loc

        if plot_error_figures:
            fig, axs = plt.subplots(2, 2)
            self.fig = fig
            self.ax_animation = axs[0][0]
            self.axs_error = np.insert(axs[1], 0, axs[0][1])
        else:
            self.fig = plt.figure()
            self.ax_animation = self.fig.add_subplot(111)

        self.ax_animation.set_title(title)
        self.line_anchor, = self.ax_animation.plot([], [], 'm-', ms=10, label="Anchor robot path")
        self.line_actual_target, = self.ax_animation.plot([], [], 'b-', ms=10, label="Actual target path")
        self.line_measured1, = self.ax_animation.plot([], [], 'g--', ms=10, label="Measured target path 1")
        self.line_measured2, = self.ax_animation.plot([], [], 'y--', ms=10, label="Measured target path 2")
        self.line_estimated, = self.ax_animation.plot([], [], 'r-', ms=10, label="Estimated target path")

        axamp = plt.axes([0.25, .03, 0.50, 0.02])
        self.slider = Slider(axamp, 'Timestep', 0, self.loc.count, valinit=0, valstep=1)
        self.is_manual = False

        self.anchor_x = self.loc.robot_system.all_anchor_positions[:, 0]
        self.anchor_y = self.loc.robot_system.all_anchor_positions[:, 1]

        self.target_x = self.loc.robot_system.all_target_positions[:, 0]
        self.target_y = self.loc.robot_system.all_target_positions[:, 1]

        self.estimated_x = self.loc.estimated_positions[:, 0]
        self.estimated_y = self.loc.estimated_positions[:, 1]

        self.measured1_x = self.loc.measured_positions[:, 0, 0]
        self.measured1_y = self.loc.measured_positions[:, 0, 1]
        self.measured2_x = self.loc.measured_positions[:, 1, 0]
        self.measured2_y = self.loc.measured_positions[:, 1, 1]

    def run(self):
        max_x = max(chain(self.anchor_x, self.target_x, self.estimated_x, self.measured1_x, self.measured2_x))
        max_y = max(chain(self.anchor_y, self.target_y, self.estimated_y, self.measured1_y, self.measured2_y))
        min_x = min(chain(self.anchor_x, self.target_x, self.estimated_x, self.measured1_x, self.measured2_x))
        min_y = min(chain(self.anchor_y, self.target_y, self.estimated_y, self.measured1_y, self.measured2_y))
        self.ax_animation.set_xlim(min_x, max_x)
        self.ax_animation.set_ylim(min_y, max_y)
        self.ax_animation.legend()

        # call update function on slider value change
        self.slider.on_changed(self.update_slider)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        ani = matplotlib.animation.FuncAnimation(self.fig, self.animate, frames=self.loc.count, interval=50,
                                                 save_count=self.loc.count)
        if self.save:
            ani.save('ani.gif', 'pillow')
        if self.plot_error_figures:
            self.error_figures()
        plt.show()

    def animate(self, frame):
        if self.is_manual:
            return self.line_anchor, self.line_actual_target, self.line_estimated,\
                   self.line_measured1, self.line_measured2

        # Calls update due to change
        self.slider.set_val(frame)
        self.is_manual = False
        return self.line_anchor, self.line_actual_target, self.line_estimated, self.line_measured1, self.line_measured2

    def update_slider(self, val):
        self.is_manual = True
        self.update(val)

    def update(self, val):
        self.line_anchor.set_data(self.anchor_x[:val], self.anchor_y[:val])
        self.line_actual_target.set_data(self.target_x[:val], self.target_y[:val])
        self.line_estimated.set_data(self.estimated_x[self.loc.idx_loc:val], self.estimated_y[self.loc.idx_loc:val])
        self.line_measured1.set_data(self.measured1_x[:val], self.measured1_y[:val])
        self.line_measured2.set_data(self.measured2_x[:val], self.measured2_y[:val])
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        # Check where the click happened
        (xm, ym), (xM, yM) = self.slider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it is handled in update_slider
            return
        else:
            # user clicked somewhere else on canvas = unpause
            self.is_manual = False

    def error_figures(self):
        pos_err = np.linalg.norm(self.loc.robot_system.all_target_positions[self.loc.idx_loc:] - self.loc.estimated_positions[self.loc.idx_loc:], axis=1)
        self.axs_error[0].plot(pos_err, label="$|| p - \hat{p} ||$")
        self.axs_error[1].plot(self.loc.robot_system.real_r, label="Real r")
        a = np.diff(self.loc.robot_system.real_r)
        self.axs_error[2].plot(np.diff(self.loc.robot_system.real_r), label="Real $\dot{r}$")

        if self.loc.robot_system.noise:
            self.axs_error[2].plot(np.diff(self.loc.robot_system.measured_r), label="Measured $\dot{r}$")
            self.axs_error[1].plot(self.loc.robot_system.measured_r, label="Measured r")
        self.axs_error[0].legend()
        self.axs_error[1].legend()
        self.axs_error[2].legend()


