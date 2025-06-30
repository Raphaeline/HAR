import subprocess
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

if __name__ == '__main__':
    subprocess.run(["python", "gui/gui_main.py"])
