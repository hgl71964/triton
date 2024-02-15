import os
import pickle
import tempfile
import time

import torch
import triton
import triton.language as tl

import random
import numpy as np

# asm (add CuAsm to PYTHONPATH!)
from CuAsm.CubinFile import CubinFile

from fgk.jit import search, jit
from fgk.autotuner import autotune as fgk_autotune
from fgk.utils.gpu_utils import get_gpu_name, get_gpu_cc

from absl import app
from absl import flags

# YAPF: disable
FLAGS = flags.FLAGS

# kernel
flags.DEFINE_string("f", None, "")
flags.DEFINE_string("save", None, "")

flags.DEFINE_integer("dump", 0, "whether to dump")
flags.DEFINE_integer("hack", 0, "whether to hack")
flags.DEFINE_integer("seed", 1337, "")
flags.DEFINE_integer("n_tests", 100, "")
flags.DEFINE_integer("n_choices", 1, "+-n choices")
flags.DEFINE_integer("load", 0, "whether to load")
flags.DEFINE_integer("bench", 0, "whether to bench")


def main(_):

    with open(FLAGS.f, 'rb') as f:
        data = pickle.load(f)
    cubin = data['cubin']

    with tempfile.NamedTemporaryFile(mode='wb', delete=True) as temp_file:

        temp_file.write(cubin)
        # Ensure data is written to the file before reading it
        temp_file.flush()
        temp_file.seek(0)

        time.sleep(1)
        cf = CubinFile(temp_file.name)

    cf.saveAsCuAsm(FLAGS.save+'.cuasm')

if __name__ == "__main__":
    app.run(main)
