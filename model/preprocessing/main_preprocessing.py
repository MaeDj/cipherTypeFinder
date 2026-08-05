###
#Launcher for preprocessing process
#Take as input dataset images and output ready-to-use data for the model described into "Unsupervised Feature Learning via Convolutional Autoencoders for
#Cross-Manuscript Comparison in Historical Cryptanalysis" from Alejandra Reinares, Giuseppe De Gregorio and Alicia Fornés
###

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.utils.progress import stage

#Anchored to this file's own directory, not the caller's cwd (main.py never
#chdir's into preprocessing/ before calling main_prepro())
SCRIPT_DIR = Path(__file__).resolve().parent


def _run_stage(script_name, binarization_method):
    #check=True: raise instead of silently continuing on a failed stage
    #binarization_method forwarded as a plain CLI arg so the stage doesn't prompt for it
    #Each stage is its own subprocess, so its own prints (including its per-image
    #progress() lines) flow straight through to this process's stdout as they happen.
    subprocess.run([sys.executable, str(SCRIPT_DIR / script_name), binarization_method], check=True)


def main_prepro(binarization_method):
    #1) Binarization process
    with stage("  1a) Binarization"):
        _run_stage("binarize.py", binarization_method)

    #2) Line segmentation process
    with stage("  1b) Line segmentation"):
        _run_stage("line_segmentation.py", binarization_method)

    #3) Cleaning process
    with stage("  1c) Cleaning"):
        _run_stage("cleaning.py", binarization_method)

    #4) Connected Component process
    with stage("  1d) Connected component extraction"):
        _run_stage("connected_component.py", binarization_method)

    #5) Processing for Clustering process
    with stage("  1e) Resize for clustering"):
        _run_stage("processing_for_clustering.py", binarization_method)