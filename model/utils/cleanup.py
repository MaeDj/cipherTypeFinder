### AI generated optimization process

###
#Shared helper for reclaiming disk space between preprocessing stages.
#Each of the five preprocessing stages (binarize -> line_segmentation -> cleaning ->
#connected_component -> processing_for_clustering) writes its own full copy of the corpus to
#disk, and by default every earlier stage's output sticks around even after the next stage has
#already consumed it. At full corpus scale (hundreds of thousands of character crops across
#hundreds of thousands of files) that's enough to exhaust a disk quota well before the run
#finishes - see processing_for_clustering.py's memmap flush, which is exactly where this was
#first hit. Once a stage has successfully produced its own output, its input (the previous
#stage's output) is never read again by anything, so it's safe to reclaim right away.
###
import os
import shutil


#Escape hatch for anyone who wants to inspect an intermediate stage's output after the fact
#(debugging one stage, comparing binarization methods side by side, etc.) - set to any value
#other than ""/"0"/"false"/"no" to keep every intermediate folder instead of deleting it.
def _keep_intermediate():
    value = os.environ.get("CIPHERTYPEFINDER_KEEP_INTERMEDIATE", "")
    return value.strip().lower() not in ("", "0", "false", "no")


def cleanup_stage_output(path, label):
    """Delete a folder that nothing downstream will read again (a previous stage's now-consumed
    output, or a side artifact like bounding-box visualizations that's never read at all).

    Never fatal: a cleanup failure (permissions, already gone, shared filesystem hiccup) is
    logged and swallowed rather than turning an otherwise-successful stage into a failed one
    over disk hygiene.
    """
    if _keep_intermediate():
        print(f"[cleanup] Keeping {label} ({path}) - CIPHERTYPEFINDER_KEEP_INTERMEDIATE is set", flush=True)
        return

    if not os.path.isdir(path):
        return

    try:
        shutil.rmtree(path)
        print(f"[cleanup] Removed now-unneeded {label}: {path}", flush=True)
    except OSError as e:
        print(f"[cleanup] Warning: could not remove {label} ({path}): {e}", flush=True)