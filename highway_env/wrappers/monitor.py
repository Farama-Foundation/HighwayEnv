from __future__ import division, print_function

import os
import datetime

from gym.wrappers import Monitor
from gym.wrappers.monitoring import video_recorder


class MonitorV2(Monitor):
    RUN_PREFIX = 'run'

    def __init__(self, env, directory, video_callable=None, force=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None):
        if video_callable is None:
            video_callable = MonitorV2.always_call_video
        directory = self.run_directory(directory)
        super(MonitorV2, self).__init__(env, directory, video_callable, force, resume, write_upon_reset, uid, mode)

    def close(self):
        self.env.close()
        super(MonitorV2, self).close()

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

        self.env.automatic_rendering_callback = self.video_recorder.capture_frame

    @staticmethod
    def always_call_video(i):
        return True

    def run_directory(self, directory):
        return os.path.join(directory, '{}_{}'.format(self.RUN_PREFIX,
                                                      datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
