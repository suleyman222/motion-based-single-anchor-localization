import matplotlib.pyplot as plt
import numpy as np


# Position Equation
def rx(t):
    return 3 + t * np.cos(t)


def ry(t):
    return t * np.sin(t)


# Length of position
def r(t):
    return np.sqrt(np.square(rx(t)) + np.square(ry(t)))


def vx(t):
    return np.cos(t) - t*np.sin(t)


def vy(t):
    return np.sin(t) + t*np.cos(t)


# Length of velocity
def s(t):
    return np.sqrt(np.square(vx(t)) + np.square(vy(t)))


# Change of r
def dr(t):
    return (rx(t)*vx(t) + ry(t)*vy(t))/r(t)


if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.gca()
    ts = np.linspace(0, 2, 100)

    # Plot velocity vectors
    t_step = 10
    for t_pos in range(0, len(ts) - 1, t_step):
        t_val_start = ts[t_pos]

        vel_start = [rx(t_val_start), ry(t_val_start)]
        vel_end = [rx(t_val_start) + vx(t_val_start), ry(t_val_start) + vy(t_val_start)]
        ax1.arrow(vel_start[0], vel_start[1], vx(t_val_start), vy(t_val_start), head_width=0.1, head_length=0.1)

    # The angles between the position and velocity vectors seem to be correct
    estimated_thetas = np.arccos(dr(ts)/s(ts))
    actual_thetas = np.array([np.arccos((vx(t) * rx(t) + vy(t) * ry(t)) / (r(t) * s(t))) for t in ts])
    print(np.sum(actual_thetas - estimated_thetas))
    # print(estimated_thetas)

    # How do we use the angles to find the position on the circle?
    # Alpha is the angle of a velocity vector
    alphas = np.arctan(vy(ts) / vx(ts))

    # Position is given by (r(t) * cos(alpha +- theta), r(t) * sin(alpha +- theta)
    est_x = r(ts) * np.cos(alphas - estimated_thetas)
    est_y = r(ts) * np.sin(alphas - estimated_thetas)

    est_x2 = r(ts) * np.cos(np.pi + estimated_thetas - alphas)
    est_y2 = r(ts) * np.sin(estimated_thetas - alphas)

    ax1.plot(rx(ts), ry(ts))
    ax1.plot(est_x, est_y)
    ax1.plot(est_x2, est_y2)
    plt.xlim(-5, 5)
    plt.ylim(0, 4)
    plt.show()
