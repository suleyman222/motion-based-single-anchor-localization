from matplotlib import pyplot as plt


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def plot_path(real_positions, measured_positions, estimated_positions):
    plt.plot(measured_positions[:, 0], measured_positions[:, 1], label='Measured position')
    plt.plot(real_positions[:, 0], real_positions[:, 1], label='Real position')
    # plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Estimated position')
    plt.xlabel('X coordinate')
    plt.xlim(0)
    plt.ylim(0)
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.show()