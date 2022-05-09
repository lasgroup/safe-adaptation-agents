from collections import defaultdict

from tensorboardX import SummaryWriter

from tensorflow import metrics


class TrainingLogger:

  def __init__(self, log_dir):
    self._writer = SummaryWriter(log_dir)
    self._metrics = defaultdict(metrics.Mean)

  def __getitem__(self, item: str):
    return self._metrics[item]

  def __setitem__(self, key: str, value: float):
    self._metrics[key].update_state(value)

  def log_summary(self, summary: dict, step: int):
    for k, v in summary.items():
      self._writer.add_scalar(k, float(v), step)
    self._writer.flush()

  def log_metrics(self, step: int):
    print("\n----Training step {} summary----".format(step))
    for k, v in self._metrics.items():
      val = float(v.result())
      print("{:<40} {:<.2f}".format(k, val))
      self._writer.add_scalar(k, val, step)
      v.reset_states()
    self._writer.flush()

  # (N, T, C, H, W)
  def log_video(self, images, step=None, name='policy', fps=30):
    self._writer.add_video(name, images, step, fps=fps)
    self._writer.flush()

  def log_figure(self, figure, step=None, name='policy'):
    self._writer.add_figure(name, figure, step)
    self._writer.flush()
