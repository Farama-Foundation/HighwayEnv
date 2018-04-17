import os

import numpy as np
import matplotlib.pyplot as plt

from highway_env.wrappers.monitor import MonitorV2

OUT_DIRECTORY = 'out'


def find_latest_run():
    files = sorted(os.listdir(OUT_DIRECTORY))
    return [os.path.join(OUT_DIRECTORY, f) for f in files if f.startswith(MonitorV2.RUN_PREFIX + '_')][-1]


def analyse(run_directory):
    print('Analysing directory', run_directory)
    manifest = MonitorV2.load_results(run_directory)
    plot(manifest['episode_rewards'], 'rewards')
    histogram(manifest['episode_rewards'], 'rewards')
    histogram(manifest['episode_avg_rewards'], 'average rewards')
    histogram(manifest['episode_lengths'], 'episode lengths')
    # scatter(manifest['episode_lengths'], manifest['episode_rewards'], 'lengths', 'rewards')


def compare(run_a, run_b):
    manifest_a = MonitorV2.load_results(run_a)
    manifest_b = MonitorV2.load_results(run_b)
    f = plot(manifest_a['episode_rewards'], 'rewards')
    plot(manifest_b['episode_rewards'], 'rewards', f)


def histogram(data, title, figure=None):
    if not figure:
        figure = plt.figure()
        plt.grid(True)
    plt.hist(data,
             weights=np.zeros_like(data) + 1. / len(data), bins=20)
    plt.title('Histogram of {}'.format(title))
    plt.xlabel(title.capitalize())
    plt.ylabel('Frequency')
    plt.show()
    return figure


def plot(data, title, figure=None):
    if not figure:
        figure = plt.figure()
        plt.grid(True)
    plt.plot(np.arange(np.size(data)), data)
    plt.title('History of {}'.format(title))
    plt.xlabel('Runs')
    plt.ylabel(title.capitalize())
    plt.show()
    return figure


def scatter(xx, yy, title_x, title_y, figure=None):
    if not figure:
        figure = plt.figure()
        plt.grid(True)
    plt.scatter(xx, yy)
    plt.title('{} with respect to {}'.format(title_x, title_y))
    plt.xlabel(title_x.capitalize())
    plt.ylabel(title_y.capitalize())
    plt.show()
    return figure


if __name__ == '__main__':
    # analyse(OUT_DIRECTORY + '/' + 'run_20180416-170328'))
    analyse(find_latest_run())
    # compare('out/run_20180416-175503', 'out/run_20180416-172942')

