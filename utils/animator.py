import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.animation


class Animator:
    def __init__(self, title: str, count, idx_loc, anchor_pos, target_pos, estimated_target_pos, measured_target_pos_1,
                 measured_target_pos_2, real_r=None, measured_r=None, save=False,
                 plot_error_figures=False):

        self.plot_error_figures = plot_error_figures
        self.save = save
        self.idx_loc = idx_loc
        self.count = count
        self.measured_r = measured_r
        self.real_r = real_r

        if plot_error_figures:
            fig, axs = plt.subplots(2, 2)
            self.fig = fig
            self.ax_animation = axs[0][0]
            self.axs_error = np.insert(axs[1], 0, axs[0][1])
        else:
            self.fig = plt.figure()
            self.ax_animation = self.fig.add_subplot(111)

        self.ax_animation.set_title(title)
        self.line_anchor, = self.ax_animation.plot(anchor_pos[0], anchor_pos[1], 'm-', ms=10, label="Anchor robot path")
        self.line_actual_target, = self.ax_animation.plot(target_pos[0], target_pos[1], 'b-', ms=10, label="Actual target path")
        self.line_measured1, = self.ax_animation.plot(measured_target_pos_1[0], measured_target_pos_1[1], 'g--', ms=10, label="Measured target path 1")
        self.line_measured2, = self.ax_animation.plot(measured_target_pos_2[0], measured_target_pos_2[1], 'y--', ms=10, label="Measured target path 2")
        self.line_estimated, = self.ax_animation.plot(estimated_target_pos[0], estimated_target_pos[1], 'r-', ms=10, label="Estimated target path")

        axamp = plt.axes([0.25, .03, 0.50, 0.02])
        self.slider = Slider(axamp, 'Timestep', 0, self.count, valinit=0, valstep=1)
        self.is_manual = False

        self.anchor_pos = anchor_pos
        self.target_pos = target_pos
        self.estimated_target_pos = estimated_target_pos
        self.measured_target_pos_1 = measured_target_pos_1
        self.measured_target_pos_2 = measured_target_pos_2

    def run(self):
        self.ax_animation.legend()

        # call update function on slider value change
        self.slider.on_changed(self._update_slider)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        ani = matplotlib.animation.FuncAnimation(self.fig, self._animate, frames=self.count, interval=50,
                                                 save_count=self.count)
        if self.save:
            ani.save('ani.gif', 'pillow')
        if self.plot_error_figures:
            self._error_figures()
        plt.show()

    def _animate(self, frame):
        if self.is_manual:
            return self.line_anchor, self.line_actual_target, self.line_estimated,\
                   self.line_measured1, self.line_measured2

        # Calls update due to change
        self.slider.set_val(frame)
        self.is_manual = False
        return self.line_anchor, self.line_actual_target, self.line_estimated, self.line_measured1, self.line_measured2

    def _update_slider(self, val):
        self.is_manual = True
        self._update(val)

    def _update(self, val):
        self.line_anchor.set_data(self.anchor_pos[0][:val], self.anchor_pos[1][:val])
        self.line_actual_target.set_data(self.target_pos[0][:val], self.target_pos[1][:val])
        self.line_estimated.set_data(self.estimated_target_pos[0][self.idx_loc:val], self.estimated_target_pos[1][self.idx_loc:val])
        self.line_measured1.set_data(self.measured_target_pos_1[0][:val], self.measured_target_pos_1[1][:val])
        self.line_measured2.set_data(self.measured_target_pos_2[0][:val], self.measured_target_pos_2[1][:val])
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        # Check where the click happened
        (xm, ym), (xM, yM) = self.slider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it is handled in update_slider
            return
        else:
            # user clicked somewhere else on canvas = unpause
            self.is_manual = False

    def _error_figures(self):
        pos_err = np.linalg.norm(self.target_pos.T[self.idx_loc:] - self.estimated_target_pos.T[self.idx_loc:], axis=1)
        self.axs_error[0].plot(pos_err, label="$|| p - \hat{p} ||$")

        self.axs_error[1].plot(self.real_r, label="Real r")
        self.axs_error[2].plot(range(1, len(self.real_r)), np.diff(self.real_r), label="Real $\dot{r}$")

        if self.measured_r:
            self.axs_error[2].plot(range(1, len(self.measured_r)), np.diff(self.measured_r), label="Measured $\dot{r}$")
            self.axs_error[1].plot(self.measured_r, label="Measured r")
        self.axs_error[0].legend()
        self.axs_error[1].legend()
        self.axs_error[2].legend()


