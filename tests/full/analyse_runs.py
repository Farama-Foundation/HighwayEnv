import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from highway_env.wrappers.monitor import MonitorV2

OUT_DIRECTORY = 'out'


def find_latest_run():
    files = sorted(os.listdir(OUT_DIRECTORY))
    runs = [f for f in files if f.startswith(MonitorV2.RUN_PREFIX + '_')]
    if not runs:
        raise FileNotFoundError("No run has been found in {}".format(OUT_DIRECTORY))
    return runs[-1:]


def analyse(runs_directories):
    runs = {directory: MonitorV2.load_results(os.path.join(OUT_DIRECTORY, directory))
            for directory in runs_directories}
    plot_all(runs, field='episode_rewards', title='rewards')
    describe_all(runs, field='episode_rewards', title='rewards')
    histogram_all(runs, field='episode_rewards', title='rewards')
    histogram_all(runs, field='episode_avg_rewards', title='average rewards')
    histogram_all(runs, field='episode_lengths', title='lengths')
    plt.show()


def compare(runs_directories_a, runs_directories_b):
    runs_a = {directory: MonitorV2.load_results(os.path.join(OUT_DIRECTORY, directory))
              for directory in runs_directories_a}
    runs_b = {directory: MonitorV2.load_results(os.path.join(OUT_DIRECTORY, directory))
              for directory in runs_directories_b}
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    plot_all(runs_a, field='episode_rewards', title='rewards', axes=ax1)
    plot_all(runs_b, field='episode_rewards', title='rewards', axes=ax2)
    plt.show()


def histogram_all(runs, field, title, axes=None):
    for directory, manifest in runs.items():
        axes = histogram(manifest[field], title=title, label=directory, axes=axes)
    axes.legend()
    axes.grid()
    return axes


def histogram(data, title, label, axes=None):
    if not axes:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.set_title('Histogram of {}'.format(title))
        axes.set_xlabel(title.capitalize())
        axes.set_ylabel('Frequency')
    axes.hist(data, weights=np.zeros_like(data) + 1. / len(data), bins=20, label=label)
    return axes


def plot_all(runs, field, title, axes=None):
    for directory, manifest in runs.items():
        axes = plot(manifest[field], title=title, label=directory, axes=axes)
    axes.legend()
    axes.grid()
    return axes


def plot(data, title, label, axes=None):
    if not axes:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.set_title('History of {}'.format(title))
        axes.set_xlabel('Runs')
        axes.set_ylabel(title.capitalize())
    axes.plot(np.arange(np.size(data)), data, label=label)
    return axes


def describe_all(runs, field, title):
    print('---', title, '---')
    for directory, manifest in runs.items():
        statistics = stats.describe(manifest[field])
        print(directory, '{:.2f} +/- {:.2f}'.format(statistics.mean, np.sqrt(statistics.variance)))


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
    # analyse(['aggressive-aggressive',
    #          'aggressive-defensive',
    #          'aggressive-robust',
    #          'defensive-aggressive',
    #          'defensive-defensive',
    #          'defensive-robust'])
    compare(['aggressive-aggressive', 'aggressive-defensive', 'aggressive-robust'],
            ['defensive-aggressive', 'defensive-defensive', 'defensive-robust'])


