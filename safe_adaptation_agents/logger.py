from tensorboardX import SummaryWriter


class TrainingLogger:

  def __init__(self, log_dir):
    self._writer = SummaryWriter(log_dir)

  def log_metrics(self, summary, step):
    print("\n----Training step {} summary----".format(step))
    for k, v in summary.items():
      val = float(v)
      print("{:<40} {:<.2f}".format(k, val))
      self._writer.add_scalar(k, val, step)
    self._writer.flush()

  # (N, T, C, H, W)
  def log_video(self, images, step=None, name='policy', fps=30):
    self._writer.add_video(name, images, step, fps=fps)
    self._writer.flush()

  def log_figure(self, figure, step=None, name='policy'):
    self._writer.add_figure(name, figure, step)
    self._writer.flush()
