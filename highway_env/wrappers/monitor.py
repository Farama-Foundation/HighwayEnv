from __future__ import division, print_function

import json
import os
import datetime

from gym import logger
from gym.wrappers import Monitor
from gym.wrappers.monitor import detect_training_manifests, collapse_env_infos, merge_stats_files
from gym.wrappers.monitoring import video_recorder
from gym.wrappers.monitoring.stats_recorder import StatsRecorder
from gym.utils import atomic_write
from gym.utils.json_utils import json_encode_np


class MonitorV2(Monitor):
    """
        A modified Environment Monitor that includes the following features:

        - Each run directory is automatically named with date and time
        - The stats recorder is a StatsRecorderV2, that implements additional features, including the environment seed
        at each episode so that they can be reproduced
        - The wrapped environment is closed with the monitor
        - Video recording of all frames during an ongoing environment step
        - Automatic saving of all fields of the stats recorder
    """
    RUN_PREFIX = 'run'
    STATS_HORIZON = 7

    def __init__(self, env, directory, add_subdirectory=True, video_callable=None, force=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None):
        if video_callable is None:
            video_callable = MonitorV2.always_call_video
        if add_subdirectory:
            directory = self.run_directory(directory)
        super(MonitorV2, self).__init__(env, directory, video_callable, force, resume, write_upon_reset, uid, mode)

    def _start(self, directory, video_callable=None, force=False, resume=False, write_upon_reset=False, uid=None,
               mode=None):
        super(MonitorV2, self)._start(directory, video_callable, force, resume, write_upon_reset, uid, mode)
        self.stats_recorder = StatsRecorderV2(self.STATS_HORIZON, directory,
                                              '{}.episode_batch.{}'.format(self.file_prefix, self.file_infix)
                                              , autoreset=self.env_semantics_autoreset, env_id=self.env.spec.id)

    def close(self):
        self.env.close()
        super(MonitorV2, self).close()

    def seed(self, seed=None):
        seeds = super(MonitorV2, self).seed(seed)
        self.stats_recorder.seed = seeds[0]  # Not sure why gym envs typically return *a list* of one seed
        return seeds

    def reset_video_recorder(self):
        # Close any existing video recorder
        if self.video_recorder:
            self._close_video_recorder()

        # Start recording the next video.
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=os.path.join(self.directory,
                                   '{}.video.{}.video{:06}'.format(self.file_prefix, self.file_infix, self.episode_id)),
            metadata={'episode_id': self.episode_id},
            enabled=self._video_enabled(),
        )

        # Instead of capturing just one frame, allow the environment to send all render frames when a step is ongoing
        if self._video_enabled():
            self.env.automatic_rendering_callback = self.video_recorder.capture_frame

    def _close_video_recorder(self):
        super(MonitorV2, self)._close_video_recorder()
        self.env.automatic_rendering_callback = None

    @staticmethod
    def always_call_video(i):
        return True

    @staticmethod
    def load_results(training_dir):
        if not os.path.exists(training_dir):
            logger.error('Training directory %s not found', training_dir)
            return

        manifests = detect_training_manifests(training_dir)
        if not manifests:
            logger.error('No manifests found in training directory %s', training_dir)
            return

        logger.debug('Uploading data from manifest %s', ', '.join(manifests))

        # Load up stats + video files
        stats_files = []
        videos = []
        env_infos = []

        for manifest in manifests:
            with open(manifest) as f:
                contents = json.load(f)
                # Make these paths absolute again
                stats_files.append(os.path.join(training_dir, contents['stats']))
                videos += [(os.path.join(training_dir, v), os.path.join(training_dir, m))
                           for v, m in contents['videos']]
                env_infos.append(contents['env_info'])

        env_info = collapse_env_infos(env_infos, training_dir)

        # If only one stats file is present, there is no need to merge and all fields are included
        if len(stats_files) == 1:
            with open(stats_files[0]) as f:
                content = json.load(f)
                content.update(
                    {'manifests': manifests,
                     'env_info': env_info,
                     'videos': videos})
                return content
        else:
            data_sources, initial_reset_timestamps, timestamps, episode_lengths, episode_rewards, episode_types, initial_reset_timestamp = merge_stats_files(
                stats_files)

            return {
                'manifests': manifests,
                'env_info': env_info,
                'data_sources': data_sources,
                'timestamps': timestamps,
                'episode_lengths': episode_lengths,
                'episode_rewards': episode_rewards,
                'episode_types': episode_types,
                'initial_reset_timestamps': initial_reset_timestamps,
                'initial_reset_timestamp': initial_reset_timestamp,
                'videos': videos,
            }

    def run_directory(self, directory):
        return os.path.join(directory, '{}_{}'.format(self.RUN_PREFIX,
                                                      datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))


class StatsRecorderV2(StatsRecorder):
    def __init__(self, horizon, directory, file_prefix, autoreset=False, env_id=None):
        super(StatsRecorderV2, self).__init__(directory, file_prefix, autoreset, env_id)

        # Average rewards
        self.horizon = horizon
        self.episode_avg_rewards = []
        self.avg_rewards = []
        self.rewards_history = []

        # Seed
        self.seed = None  # Set by the monitor when seeding the wrapped env
        self.episode_seeds = []

    def after_reset(self, observation):
        self.rewards_history = []
        self.avg_rewards = []
        super(StatsRecorderV2, self).after_reset(observation)

    def after_step(self, observation, reward, done, info):
        # Aggregate rewards history
        self.rewards_history.append(reward)
        if len(self.rewards_history) > self.horizon:
            self.rewards_history = self.rewards_history[:-self.horizon]
        # Compute running average
        self.avg_rewards.append(sum(self.rewards_history) / len(self.rewards_history))

        super(StatsRecorderV2, self).after_step(observation, reward, done, info)

    def save_complete(self):
        if self.steps is not None:
            self.episode_avg_rewards.append(sum(self.avg_rewards) / len(self.avg_rewards))
            self.episode_seeds.append(self.seed)
            super(StatsRecorderV2, self).save_complete()

    def flush(self):
        if self.closed:
            return

        with atomic_write.atomic_write(self.path) as f:
            json.dump({
                'initial_reset_timestamp': self.initial_reset_timestamp,
                'timestamps': self.timestamps,
                'episode_lengths': self.episode_lengths,
                'episode_rewards': self.episode_rewards,
                'episode_avg_rewards': self.episode_avg_rewards,
                'episode_seeds': self.episode_seeds,
                'episode_types': self.episode_types,
            }, f, default=json_encode_np)
