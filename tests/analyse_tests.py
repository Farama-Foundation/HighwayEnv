import numpy as np
import matplotlib.pyplot as plt

from gym.wrappers.monitor import load_results


def analyse(run_directory):
    manifest = load_results(run_directory)
    histogram(manifest['episode_rewards'], 'rewards')
    histogram(manifest['episode_lengths'], 'episode lengths')
    plot(manifest['episode_rewards'], 'rewards')
    scatter(manifest['episode_lengths'], manifest['episode_rewards'], 'lengths', 'rewards')


def histogram(data, title):
    plt.grid(True)
    plt.hist(data,
             weights=np.zeros_like(data) + 1. / len(data), bins=20)
    plt.title('Histogram of {}'.format(title))
    plt.xlabel(title.capitalize())
    plt.ylabel('Frequency')
    plt.show()


def plot(data, title):
    plt.grid(True)
    plt.plot(np.arange(np.size(data)), data)
    plt.title('History of {}'.format(title))
    plt.xlabel(title.capitalize())
    plt.ylabel('Frequency')
    plt.show()


def scatter(xx, yy, title_x, title_y):
    plt.grid(True)
    plt.scatter(xx, yy)
    plt.title('{} with respect to {}'.format(title_x, title_y))
    plt.xlabel(title_x.capitalize())
    plt.ylabel(title_y.capitalize())
    plt.show()


if __name__ == '__main__':
    analyse('out/run_20180413-171104')
