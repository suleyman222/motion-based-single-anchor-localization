import numpy as np
from matplotlib import pyplot as plt


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def plot_path(real_positions, measured_positions, estimated_positions, title):
    plt.plot(measured_positions[:, 0], measured_positions[:, 1], label='Measured position')
    plt.plot(real_positions[:, 0], real_positions[:, 1], label='Real position')
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Estimated position')
    plt.title(title)
    plt.xlabel('X coordinate')
    # plt.xlim(0)
    # plt.ylim(0)
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.show()


def rmse(measured_positions, real_positions):
    return np.sqrt(np.sum((real_positions - measured_positions)**2, axis=0) / len(real_positions))


def cos_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def closest_to(target, options):
    distance = float("inf")
    closest = None
    for option in options:
        d = np.linalg.norm([a - b for a, b in zip(option, target)])
        if d < distance:
            closest = option
            distance = d
    return closest
