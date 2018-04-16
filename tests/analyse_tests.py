import os

import numpy as np
import matplotlib.pyplot as plt

from gym.wrappers.monitor import load_results

from highway_env.wrappers.monitor import MonitorV2

OUT_DIRECTORY = 'out'


def find_latest_run():
    files = sorted(os.listdir(OUT_DIRECTORY))
    return [os.path.join(OUT_DIRECTORY, f) for f in files if f.startswith(MonitorV2.RUN_PREFIX + '_')][-1]


def analyse(run_directory):
    print('Analysing directory', run_directory)
    manifest = load_results(run_directory)
    plot(manifest['episode_rewards'], 'rewards')
    histogram(manifest['episode_rewards'], 'rewards')
    histogram(manifest['episode_lengths'], 'episode lengths')
    # scatter(manifest['episode_lengths'], manifest['episode_rewards'], 'lengths', 'rewards')


def histogram(data, title):
    plt.figure()
    plt.grid(True)
    plt.hist(data,
             weights=np.zeros_like(data) + 1. / len(data), bins=20)
    plt.title('Histogram of {}'.format(title))
    plt.xlabel(title.capitalize())
    plt.ylabel('Frequency')
    plt.show()


def plot(data, title):
    plt.figure()
    plt.grid(True)
    plt.plot(np.arange(np.size(data)), data)
    plt.title('History of {}'.format(title))
    plt.xlabel('Runs')
    plt.ylabel(title.capitalize())
    plt.show()


def scatter(xx, yy, title_x, title_y):
    plt.figure()
    plt.grid(True)
    plt.scatter(xx, yy)
    plt.title('{} with respect to {}'.format(title_x, title_y))
    plt.xlabel(title_x.capitalize())
    plt.ylabel(title_y.capitalize())
    plt.show()


if __name__ == '__main__':
    # analyse('{}/{}'.format(OUT_DIRECTORY, 'run_20180416-170328'))
    analyse(find_latest_run())

