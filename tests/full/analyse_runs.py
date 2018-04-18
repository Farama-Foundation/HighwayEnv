import os

import numpy as np
import matplotlib.pyplot as plt

from highway_env.wrappers.monitor import MonitorV2

OUT_DIRECTORY = 'out'


def find_latest_run():
    files = sorted(os.listdir(OUT_DIRECTORY))
    runs = [os.path.join(OUT_DIRECTORY, f) for f in files if f.startswith(MonitorV2.RUN_PREFIX + '_')]
    if not runs:
        raise FileNotFoundError("No run has been found in {}".format(OUT_DIRECTORY))
    return runs[-1:]


def analyse(runs_directories):
    runs = {directory: MonitorV2.load_results(directory) for directory in runs_directories}
    plot_all(runs, field='episode_rewards', title='rewards')
    histogram_all(runs, field='episode_rewards', title='rewards')
    histogram_all(runs, field='episode_avg_rewards', title='average rewards')
    histogram_all(runs, field='episode_lengths', title='lengths')
    plt.show()


def histogram_all(runs, field, title, figure=None):
    for directory, manifest in runs.items():
        figure = histogram(manifest[field], title=title, label=directory, figure=figure)
    plt.legend()


def histogram(data, title, label, figure=None):
    if not figure:
        figure = plt.figure()
        plt.grid(True)
        plt.title('Histogram of {}'.format(title))
        plt.xlabel(title.capitalize())
        plt.ylabel('Frequency')
    plt.hist(data, weights=np.zeros_like(data) + 1. / len(data), bins=20, label=label)
    return figure


def plot_all(runs, field, title, figure=None):
    for directory, manifest in runs.items():
        figure = plot(manifest[field], title=title, label=directory, figure=figure)
    plt.legend()


def plot(data, title, label, figure=None):
    if not figure:
        figure = plt.figure()
        plt.grid(True)
        plt.title('History of {}'.format(title))
        plt.xlabel('Runs')
        plt.ylabel(title.capitalize())
    plt.plot(np.arange(np.size(data)), data, label=label)
    return figure


def scatter(xx, yy, title_x, title_y, label, figure=None):
    if not figure:
        figure = plt.figure()
        plt.grid(True)
    plt.scatter(xx, yy, label=label)
    plt.title('{} with respect to {}'.format(title_x, title_y))
    plt.xlabel(title_x.capitalize())
    plt.ylabel(title_y.capitalize())
    plt.show()
    return figure


if __name__ == '__main__':
    # analyse(find_latest_run())
    analyse(['out/linear-linear', 'out/idm-idm', 'out/linear-idm', 'out/idm-linear'])

