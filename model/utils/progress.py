###
#Lightweight timing/progress helpers shared across the pipeline.
#Every print here is flush=True on purpose: when the pipeline runs on a remote server
#under nohup/tmux and its stdout is redirected to a log file, Python buffers stdout by
#default whenever it isn't a tty - without flush=True nothing shows up until the buffer
#fills or the process exits, defeating the point of a "live" progress readout.
###
import time
from contextlib import contextmanager


def _ts():
    return time.strftime("%H:%M:%S")


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def log(msg):
    print(msg, flush=True)


@contextmanager
def stage(name):
    """Wrap a pipeline stage: logs a start line, then a done/failed line with elapsed time."""
    start = time.monotonic()
    log(f"[{_ts()}] [START] {name}")
    try:
        yield
    except Exception:
        log(f"[{_ts()}] [FAIL ] {name} - after {format_duration(time.monotonic() - start)}")
        raise
    else:
        log(f"[{_ts()}] [DONE ] {name} - took {format_duration(time.monotonic() - start)}")


def progress(iterable, total=None, label="", every=25, min_interval=5.0):
    """Drop-in wrapper around a for-loop that periodically logs count/ETA/rate.

    Logs on whichever comes first: every `every` items, or every `min_interval` seconds,
    plus always on the last item - so slow per-item loops still get regular updates and
    fast ones don't spam the log.
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    start = time.monotonic()
    last = start
    count = 0
    for item in iterable:
        count += 1
        yield item
        now = time.monotonic()
        due = (count % every == 0) or (now - last >= min_interval) or (count == total)
        if due:
            elapsed = now - start
            rate = count / elapsed if elapsed > 0 else 0
            if total:
                eta = (total - count) / rate if rate > 0 else 0
                pct = 100 * count / total
                log(f"    [{label}] {count}/{total} ({pct:.1f}%) - elapsed {format_duration(elapsed)} "
                    f"- ETA {format_duration(eta)} - {rate:.2f}/s")
            else:
                log(f"    [{label}] {count} done - elapsed {format_duration(elapsed)} - {rate:.2f}/s")
            last = now