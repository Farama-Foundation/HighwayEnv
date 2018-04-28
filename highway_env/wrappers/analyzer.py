import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from highway_env.wrappers.monitor import MonitorV2


class RunAnalyzer(object):
    def __init__(self, directory, runs=None):
        self.directory = directory

        if not runs:
            runs = self.find_latest_run()

        self.analyse(runs)

    def find_latest_run(self):
        files = sorted(os.listdir(self.directory))
        runs = [f for f in files if f.startswith(MonitorV2.RUN_PREFIX + '_')]
        if not runs:
            raise FileNotFoundError("No run has been found in {}".format(self.directory))
        return runs[-1:]

    def analyse(self, runs_directories):
        runs = {directory: MonitorV2.load_results(os.path.join(self.directory, directory))
                for directory in runs_directories}
        self.plot_all(runs, field='episode_rewards', title='rewards')
        self.describe_all(runs, field='episode_rewards', title='rewards')
        self.histogram_all(runs, field='episode_rewards', title='rewards')
        # self.histogram_all(runs, field='episode_avg_rewards', title='average rewards')
        self.histogram_all(runs, field='episode_lengths', title='lengths')
        plt.show()

    def compare(self, runs_directories_a, runs_directories_b):
        runs_a = {directory: MonitorV2.load_results(os.path.join(self.directory, directory))
                  for directory in runs_directories_a}
        runs_b = {directory: MonitorV2.load_results(os.path.join(self.directory, directory))
                  for directory in runs_directories_b}
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        self.plot_all(runs_a, field='episode_rewards', title='rewards', axes=ax1)
        self.plot_all(runs_b, field='episode_rewards', title='rewards', axes=ax2)
        plt.show()

    def histogram_all(self, runs, field, title, axes=None):
        dirs = list(runs.keys())
        data = [runs[directory][field] for directory in dirs]
        axes = self.histogram(data, title=title, label=dirs, axes=axes)
        axes.legend()
        axes.grid()
        return axes

    def histogram(self, data, title, label, axes=None):
        if not axes:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.set_title('Histogram of {}'.format(title))
            axes.set_xlabel(title.capitalize())
            axes.set_ylabel('Count')
        axes.hist(data, label=label)
        return axes

    def plot_all(self, runs, field, title, axes=None):
        for directory, manifest in runs.items():
            axes = self.plot(manifest[field], title=title, label=directory, axes=axes)
        axes.legend()
        axes.grid()
        return axes

    def plot(self, data, title, label, axes=None):
        if not axes:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.set_title('History of {}'.format(title))
            axes.set_xlabel('Runs')
            axes.set_ylabel(title.capitalize())
        axes.plot(np.arange(np.size(data)), data, label=label)
        return axes

    def describe_all(self, runs, field, title):
        print('---', title, '---')
        for directory, manifest in runs.items():
            statistics = stats.describe(manifest[field])
            print(directory, '{:.2f} +/- {:.2f}'.format(statistics.mean, np.sqrt(statistics.variance)))

    def scatter(self, xx, yy, title_x, title_y, label, figure=None):
        if not figure:
            figure = plt.figure()
            plt.grid(True)
        plt.scatter(xx, yy, label=label)
        plt.title('{} with respect to {}'.format(title_x, title_y))
        plt.xlabel(title_x.capitalize())
        plt.ylabel(title_y.capitalize())
        plt.show()
        return figure