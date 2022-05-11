import os

from queue import Queue
from threading import Thread

from typing import Optional

from collections import defaultdict

import cloudpickle

from tensorboardX import SummaryWriter

from tensorflow import metrics


class TrainingLogger:

  def __init__(self, log_dir):
    self._writer = SummaryWriter(log_dir)
    self._metrics = defaultdict(metrics.Mean)
    self.step = 0
    self.log_dir = log_dir

  def __getitem__(self, item: str):
    return self._metrics[item]

  def __setitem__(self, key: str, value: float):
    self._metrics[key].update_state(value)

  def log_summary(self, summary: dict, step: Optional[int] = None):
    step = step or self.step
    for k, v in summary.items():
      self._writer.add_scalar(k, float(v), step)
    self._writer.flush()

  def log_metrics(self, step: Optional[int] = None):
    step = step or self.step
    print("\n----Training step {} summary----".format(step))
    for k, v in self._metrics.items():
      val = float(v.result())
      print("{:<40} {:<.2f}".format(k, val))
      self._writer.add_scalar(k, val, step)
      v.reset_states()
    self._writer.flush()

  # (N, T, C, H, W)
  def log_video(self, images, name='policy', fps=30):
    self._writer.add_video(name, images, self.step, fps=fps)
    self._writer.flush()

  def log_figure(self, figure, name='policy'):
    self._writer.add_figure(name, figure, self.step)
    self._writer.flush()

  def __getstate__(self):
    self._writer.close()
    self._metrics.clear()
    state = self.__dict__.copy()
    del state['_writer']
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._writer = SummaryWriter(self.log_dir)


class StateWriter:

  def __init__(self, log_dir: str):
    self._file_handle = None
    self.log_dir = log_dir
    self.queue = Queue(maxsize=5)
    self._thread = Thread(name="state_writer", target=self._worker)
    self._thread.start()

  def write(self, data: dict):
    self.queue.put(data)
    if not self._thread.is_alive():
      self._thread = Thread(name="state_writer", target=self._worker)
      self._thread.start()

  def _worker(self):
    while not self.queue.empty():
      data = self.queue.get()
      if self._file_handle is None:
        self._file_handle = open(os.path.join(self.log_dir, 'state.pkl'), 'wb')
      cloudpickle.dump(data, self._file_handle)
      self.queue.task_done()

  def close(self):
    self.queue.join()
    self._thread.join()
    if self._file_handle is not None:
      self._file_handle.close()
