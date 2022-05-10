from typing import Optional

from collections import defaultdict

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

  def log_summary(self, summary: dict):
    for k, v in summary.items():
      self._writer.add_scalar(k, float(v), self.step)
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
    """
    Define how the agent should be pickled.
    """
    state = self.__dict__.copy()
    del state['_writer']
    return state

  def __setstate__(self, state):
    """
    Define how the agent should be loaded.
    """
    self.__dict__.update(state)
    self._writer = SummaryWriter(self.log_dir)
