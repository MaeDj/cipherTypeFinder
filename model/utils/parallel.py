###
#Shared process-pool helper for the preprocessing stages (binarize/line_segmentation/cleaning/
#connected_component/processing_for_clustering). Every one of those stages processes hundreds to
#hundreds of thousands of files that are fully independent of each other, one at a time in a plain
#Python for-loop - on a many-core server (e.g. the Caramba lab's dual-EPYC PowerEdge R7525) that
#leaves almost every core idle. Routing that per-file work through a process pool is the single
#biggest lever available for these stages' wall-clock time.
###
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def default_workers():
    #Leave one core free for the OS/orchestrating process rather than saturating every thread -
    #still uses effectively all cores on a many-core server.
    cpu = os.cpu_count() or 1
    return max(1, cpu - 1)


def _init_worker():
    #cv2 parallelizes many of its own ops internally (parallel_for_) using a thread pool per
    #process. Combined with process-level parallelism here, that oversubscribes the machine's
    #cores (N worker processes x M internal cv2 threads each) and makes things slower, not
    #faster. Each worker does exactly one image at a time, so it gets its own core from the
    #process pool and doesn't need OpenCV's internal threading on top of that.
    import cv2
    cv2.setNumThreads(1)


def parallel_map_unordered(fn, items, max_workers=None):
    """Run fn(item) for every item across a process pool, yielding (item, result) pairs as they
    complete - not in submission order - so a live progress readout isn't blocked waiting on
    whichever item happens to be slowest. `fn` must be a module-level function (picklable) and
    rely only on module globals already set before the pool is created (safe under the default
    fork start method on Linux, where workers inherit the parent's already-initialized state)."""
    items = list(items)
    if not items:
        return
    max_workers = max_workers or default_workers()
    #A handful of files doesn't justify pool startup overhead.
    if max_workers <= 1 or len(items) <= 1:
        for item in items:
            yield item, fn(item)
        return

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            yield futures[future], future.result()