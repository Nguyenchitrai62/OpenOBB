"""Compact training console: one in-place progress row per epoch and a validation row, in the familiar
'Epoch / GPU_mem / losses / Instances / Size' layout. Detailed logs go to files, not here."""
import sys
import time

COLS = ("cls_loss", "box_loss", "kld_loss", "angle_loss", "dfl_loss")
KEYS = ("loss_mal", "loss_bbox", "loss_kld", "loss_angle", "loss_fgl")      # final decoder layer losses


def _bar(i, n, width=12):
    k = int(width * i / max(n, 1))
    return "━" * k + "─" * (width - k)


def _rate(i, dt):
    if i <= 0 or dt <= 0:
        return ""
    r = i / dt
    return f"{r:.1f}it/s" if r >= 1 else f"{1 / r:.1f}s/it"


def _dur(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}:{int(s % 60):02d}"


class EpochBar:
    """Usage: bar = EpochBar(epoch, epochs, n_iters, size); bar.update(i, means, inst, mem) ...; bar.close()."""

    def __init__(self, epoch, epochs, n, size, stream=sys.stdout, every=1.0):
        self.epoch, self.epochs, self.n, self.size = epoch, epochs, n, size
        self.stream, self.every = stream, every
        self.t0 = self.last = time.time()
        print("\n" + f"{'Epoch':>11}{'GPU_mem':>11}" + "".join(f"{c:>11}" for c in COLS)
              + f"{'Instances':>11}{'Size':>11}", file=stream, flush=True)

    def update(self, i, means, instances, mem_gb, force=False):
        now = time.time()
        if not force and now - self.last < self.every and i < self.n:
            return
        self.last = now
        vals = "".join(f"{means.get(k, 0.0):>11.4f}" for k in KEYS)
        mem = f"{mem_gb:.1f}G" if mem_gb is not None else "-"
        self.row = (f"{f'{self.epoch + 1}/{self.epochs}':>11}{mem:>11}{vals}{instances:>11}{self.size:>11}: "
                    f"{100 * i / max(self.n, 1):3.0f}% {_bar(i, self.n)} {i}/{self.n} {_rate(i, now - self.t0)} "
                    f"{_dur(now - self.t0)}")
        self.stream.write("\r" + self.row)
        self.stream.flush()

    def close(self):
        self.stream.write("\n")
        self.stream.flush()


def val_rows(res, seconds=None, stream=sys.stdout):
    """'Class Images Instances P R mAP50 mAP50-95' header + the 'all' row."""
    from vrdet.eval.dota import summary_table
    head, row = summary_table(res, per_class=False).splitlines()
    print(head + (f"  ({_dur(seconds)})" if seconds is not None else ""), file=stream)
    print(row, file=stream, flush=True)
